#!/usr/bin/env bash
# Run Qwen2.5-Coder sizes on ViSpider using only the schema-description prompt
# augmentation file. This does not run contextual augmentation and does not use
# precomputed hints.

set -euo pipefail

DATASET="${DATASET:-vispider}"
SPLIT="${SPLIT:-dev}"
SIZES="${SIZES:-0_5B 1_5B 3B 7B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
LIMIT="${LIMIT:-}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-descriptions/db_descriptions/schema_description_20db.json}"
RESULT_ROOT="${RESULT_ROOT:-zero_shot/results/qwen_schema_description/${SPLIT}}"
PRIMARY_METRIC="${PRIMARY_METRIC:-execution_accuracy}"

model_id_for() {
  case "$1" in
    0_5B) echo "Qwen/Qwen2.5-Coder-0.5B-Instruct" ;;
    1_5B) echo "Qwen/Qwen2.5-Coder-1.5B-Instruct" ;;
    3B)   echo "Qwen/Qwen2.5-Coder-3B-Instruct" ;;
    7B)   echo "Qwen/Qwen2.5-Coder-7B-Instruct" ;;
    14B)  echo "Qwen/Qwen2.5-Coder-14B-Instruct" ;;
    32B)  echo "Qwen/Qwen2.5-Coder-32B-Instruct" ;;
    *)    echo "" ;;
  esac
}

run_one() {
  local size="$1"
  local model_id="$2"
  local outdir="${RESULT_ROOT}/${size}"

  mkdir -p "$outdir"

  local pred_path="${outdir}/predictions.txt"
  local gold_path="${outdir}/gold.txt"
  local em_path="${outdir}/eval_em_only.json"
  local ex_path="${outdir}/eval_ex.json"
  local details_path="${outdir}/exec_details.json"
  local log_path="${outdir}/run.log"

  echo "============================================================"
  echo " [schema-description] Running ${size}: ${model_id}"
  echo " mode        -> schema descriptions only"
  echo " descriptions -> ${DESCRIPTION_FILE}"
  echo " hints       -> disabled"
  echo " out         -> ${outdir}"
  echo "============================================================"

  local inf_cmd=(python -m zero_shot.run_zero_shot
    --mode inference
    --dataset "$DATASET"
    --split "$SPLIT"
    --model "$model_id"
    --descriptions-file "$DESCRIPTION_FILE"
    --output "$em_path"
    --predictions-output "$pred_path"
    --gold-output "$gold_path"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL")

  if [[ -n "$LIMIT" ]]; then
    inf_cmd+=(--limit "$LIMIT")
  fi

  echo "[Phase 1] Inference + EM (schema-description)" | tee "$log_path"
  if ! "${inf_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! schema-description ${size} Phase 1 failed. See ${log_path}" | tee -a "$log_path"
    return 0
  fi

  local ex_cmd=(python -m zero_shot.run_zero_shot
    --mode exec
    --dataset "$DATASET"
    --split "$SPLIT"
    --model "$model_id"
    --predictions-input "$pred_path"
    --gold-input "$gold_path"
    --em-input "$em_path"
    --output "$ex_path"
    --exec-details-output "$details_path"
    --timeout-seconds "$TIMEOUT_SECONDS")

  echo "" | tee -a "$log_path"
  echo "[Phase 2] Execution accuracy (schema-description)" | tee -a "$log_path"
  if ! "${ex_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! schema-description ${size} Phase 2 failed. Phase 1 artifacts preserved in ${outdir}" | tee -a "$log_path"
  fi
}

if [[ ! -f "$DESCRIPTION_FILE" ]]; then
  echo "!! Missing schema description file: ${DESCRIPTION_FILE}"
  exit 1
fi

mkdir -p "$RESULT_ROOT"

for size in $SIZES; do
  model_id="$(model_id_for "$size")"
  if [[ -z "$model_id" ]]; then
    echo "!! Unknown size tag: ${size} (skipping)"
    continue
  fi
  run_one "$size" "$model_id"
done

SUMMARY_LOG="${RESULT_ROOT}/summary.log"
SUMMARY_JSON="${RESULT_ROOT}/summary.json"

DESCRIPTION_FILE="$DESCRIPTION_FILE" python - "$RESULT_ROOT" "$PRIMARY_METRIC" "$SUMMARY_JSON" <<'PY' | tee "$SUMMARY_LOG"
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
primary_metric = sys.argv[2]
summary_json = Path(sys.argv[3])
description_file = os.environ.get("DESCRIPTION_FILE")

rows = []
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir():
        continue
    ex_file = model_dir / "eval_ex.json"
    em_file = model_dir / "eval_em_only.json"
    if ex_file.exists():
        data = json.loads(ex_file.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        em_summary = data.get("em_summary", {})
    elif em_file.exists():
        data = json.loads(em_file.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        em_summary = summary
    else:
        continue
    db_desc = (em_summary or summary).get("database_descriptions", {})
    rows.append(
        {
            "model": model_dir.name,
            "count": summary.get("count", 0),
            "exact_match": summary.get("exact_match"),
            "execution_accuracy": summary.get("execution_accuracy"),
            "description_file": db_desc.get("descriptions_file") or description_file,
            "examples_with_descriptions": db_desc.get("examples_with_descriptions"),
        }
    )


def metric_key(row):
    primary = row.get(primary_metric)
    secondary_name = "exact_match" if primary_metric == "execution_accuracy" else "execution_accuracy"
    secondary = row.get(secondary_name)
    return (
        primary if primary is not None else -1,
        secondary if secondary is not None else -1,
    )


best = max(rows, key=metric_key) if rows else None
summary_json.write_text(
    json.dumps(
        {
            "run": "schema_description",
            "primary_metric": primary_metric,
            "description_file": description_file,
            "best": best,
            "rows": rows,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


def fmt(value):
    return f"{value:.4f}" if value is not None else "n/a"


print("=== Schema-description summary ===")
print(f"primary metric: {primary_metric}")
print(f"description file: {description_file}")
print(f"{'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}  {'desc_N':>7}")
for row in rows:
    desc_n = row.get("examples_with_descriptions")
    desc_text = str(desc_n) if desc_n is not None else "n/a"
    print(
        f"{row['model']:>8}  {row['count']:>5}  "
        f"{fmt(row['exact_match']):>8}  {fmt(row['execution_accuracy']):>8}  "
        f"{desc_text:>7}"
    )

if best:
    print()
    print(
        "Best: "
        f"model={best['model']}, "
        f"EM={fmt(best['exact_match'])}, EX={fmt(best['execution_accuracy'])}"
    )
print(f"Summary JSON -> {summary_json}")
PY

echo "Summary saved to ${SUMMARY_LOG}"
