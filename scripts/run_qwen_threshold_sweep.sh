#!/usr/bin/env bash
# Run the same zero-shot pipeline with precomputed augmentation hints for
# multiple thresholds, then summarize the best threshold/model combination.

set -euo pipefail

DATASET="${DATASET:-vispider}"
SPLIT="${SPLIT:-dev}"
SIZES="${SIZES:-0_5B 1_5B 3B 7B}"
THRESHOLDS="${THRESHOLDS:-0.40 0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
LIMIT="${LIMIT:-}"
HINTS_ROOT="${HINTS_ROOT:-augmentation/results_embeddinggemma}"
RESULT_ROOT="${RESULT_ROOT:-zero_shot/results/qwen_threshold_sweep/${SPLIT}}"
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

threshold_tag() {
  local normalized
  normalized="$(printf "%.2f" "$1")"
  echo "t${normalized/./}"
}

run_one() {
  local threshold="$1"
  local tag="$2"
  local size="$3"
  local model_id="$4"
  local hints_path="${HINTS_ROOT}/${SPLIT}_${tag}/hints.json"
  local outdir="${RESULT_ROOT}/${tag}/${size}"

  if [[ ! -f "$hints_path" ]]; then
    echo "!! Missing hints file: ${hints_path}"
    return 1
  fi

  mkdir -p "$outdir"

  local pred_path="${outdir}/predictions.txt"
  local gold_path="${outdir}/gold.txt"
  local em_path="${outdir}/eval_em_only.json"
  local ex_path="${outdir}/eval_ex.json"
  local details_path="${outdir}/exec_details.json"
  local log_path="${outdir}/run.log"

  echo "============================================================"
  echo " [threshold=${threshold}] Running ${size}: ${model_id}"
  echo " mode  -> precomputed hints (--hints-input only)"
  echo " hints -> ${hints_path}"
  echo " out   -> ${outdir}"
  echo "============================================================"

  local inf_cmd=(python -m zero_shot.run_zero_shot
    --mode inference
    --dataset "$DATASET"
    --split "$SPLIT"
    --model "$model_id"
    --hints-input "$hints_path"
    --augment-threshold "$threshold"
    --output "$em_path"
    --predictions-output "$pred_path"
    --gold-output "$gold_path"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL")

  if [[ -n "$LIMIT" ]]; then
    inf_cmd+=(--limit "$LIMIT")
  fi

  echo "[Phase 1] Inference + EM" | tee "$log_path"
  if ! "${inf_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! threshold=${threshold} ${size} Phase 1 failed. See ${log_path}" | tee -a "$log_path"
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
  echo "[Phase 2] Execution accuracy" | tee -a "$log_path"
  if ! "${ex_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! threshold=${threshold} ${size} Phase 2 failed. Phase 1 artifacts preserved in ${outdir}" | tee -a "$log_path"
  fi
}

mkdir -p "$RESULT_ROOT"

for threshold in $THRESHOLDS; do
  tag="$(threshold_tag "$threshold")"
  for size in $SIZES; do
    model_id="$(model_id_for "$size")"
    if [[ -z "$model_id" ]]; then
      echo "!! Unknown size tag: ${size} (skipping)"
      continue
    fi
    run_one "$threshold" "$tag" "$size" "$model_id"
  done
done

SUMMARY_LOG="${RESULT_ROOT}/summary.log"
SUMMARY_JSON="${RESULT_ROOT}/summary.json"

python - "$RESULT_ROOT" "$PRIMARY_METRIC" "$SUMMARY_JSON" <<'PY' | tee "$SUMMARY_LOG"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
primary_metric = sys.argv[2]
summary_json = Path(sys.argv[3])

rows = []
for threshold_dir in sorted(root.iterdir()):
    if not threshold_dir.is_dir():
        continue
    for model_dir in sorted(threshold_dir.iterdir()):
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
        augmentation = (em_summary or summary).get("augmentation", {})
        rows.append(
            {
                "threshold": threshold_dir.name,
                "model": model_dir.name,
                "count": summary.get("count", 0),
                "exact_match": summary.get("exact_match"),
                "execution_accuracy": summary.get("execution_accuracy"),
                "augmentation_mode": augmentation.get("mode"),
                "hints_input": augmentation.get("hints_input"),
                "examples_with_hints": augmentation.get("examples_with_hints"),
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
            "primary_metric": primary_metric,
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

print("=== Threshold sweep summary ===")
print(f"primary metric: {primary_metric}")
print(f"{'threshold':>10}  {'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}")
for row in rows:
    print(
        f"{row['threshold']:>10}  {row['model']:>8}  {row['count']:>5}  "
        f"{fmt(row['exact_match']):>8}  {fmt(row['execution_accuracy']):>8}"
    )

if best:
    print()
    print(
        "Best: "
        f"threshold={best['threshold']}, model={best['model']}, "
        f"EM={fmt(best['exact_match'])}, EX={fmt(best['execution_accuracy'])}"
    )
print(f"Summary JSON -> {summary_json}")
PY

echo "Summary saved to ${SUMMARY_LOG}"
