#!/usr/bin/env bash
# Compare Qwen2.5-Coder sizes on ViSpider with two independent phases per run:
#   Phase 1: inference + EM only, saves predictions/gold/eval_em_only.json
#   Phase 2: execution accuracy from saved artifacts, saves exec_details/eval_ex.json
#
# Runs baseline and (optionally) augmentation in parallel result trees:
#   zero_shot/results/qwen_compare/<size>/
#   zero_shot/results/qwen_compare_aug/<size>/

set -euo pipefail

DATASET="${DATASET:-vispider}"
SPLIT="${SPLIT:-dev}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LIMIT="${LIMIT:-}"
SIZES="${SIZES:-0_5B 1_5B 3B 7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
AUGMENT_THRESHOLD="${AUGMENT_THRESHOLD:-0.4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_AUG="${RUN_AUG:-1}"
HINTS_ROOT="${HINTS_ROOT:-augmentation/results_embeddinggemma}"
AUGMENT_HINTS_INPUT="${AUGMENT_HINTS_INPUT:-}"

BASELINE_ROOT="zero_shot/results/qwen_compare"
AUG_ROOT="zero_shot/results/qwen_compare_aug"

model_id_for() {
  case "$1" in
    0_5B) echo "Qwen/Qwen2.5-Coder-0.5B-Instruct" ;;
    1_5B) echo "Qwen/Qwen2.5-Coder-1.5B-Instruct" ;;
    3B)   echo "Qwen/Qwen2.5-Coder-3B-Instruct"   ;;
    7B)   echo "Qwen/Qwen2.5-Coder-7B-Instruct"   ;;
    14B)  echo "Qwen/Qwen2.5-Coder-14B-Instruct"  ;;
    32B)  echo "Qwen/Qwen2.5-Coder-32B-Instruct"  ;;
    *)    echo "" ;;
  esac
}

threshold_tag() {
  local normalized
  normalized="$(printf "%.2f" "$1")"
  echo "t${normalized/./}"
}

run_model_variant() {
  local variant_label="$1"
  local variant_root="$2"
  local augment_enabled="$3"
  local tag="$4"
  local model_id="$5"

  local outdir="${variant_root}/${tag}"
  mkdir -p "$outdir"

  local pred_path="${outdir}/predictions.txt"
  local gold_path="${outdir}/gold.txt"
  local em_path="${outdir}/eval_em_only.json"
  local ex_path="${outdir}/eval_ex.json"
  local details_path="${outdir}/exec_details.json"
  local log_path="${outdir}/run.log"

  echo "============================================================"
  echo " [${variant_label}] Running ${tag}: ${model_id}"
  echo " -> ${outdir}"
  echo "============================================================"

  local hints_path=""
  if [[ "$augment_enabled" == "1" ]]; then
    hints_path="$AUGMENT_HINTS_INPUT"
    if [[ -z "$hints_path" ]]; then
      hints_path="${HINTS_ROOT}/${SPLIT}_$(threshold_tag "$AUGMENT_THRESHOLD")/hints.json"
    fi
    if [[ ! -f "$hints_path" ]]; then
      echo "!! Missing precomputed hints file: ${hints_path}"
      echo "   Create it with python -m augmentation.run_augment, or set AUGMENT_HINTS_INPUT."
      return 1
    fi
    echo " mode  -> precomputed hints (--hints-input only)"
    echo " hints -> ${hints_path}"
  fi

  local inf_cmd=(python -m zero_shot.run_zero_shot
        --mode inference
        --dataset "$DATASET"
        --split "$SPLIT"
        --model "$model_id"
        --output "$em_path"
        --predictions-output "$pred_path"
        --gold-output "$gold_path"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL")

  if [[ -n "$LIMIT" ]]; then
    inf_cmd+=(--limit "$LIMIT")
  fi

  if [[ "$augment_enabled" == "1" ]]; then
    inf_cmd+=(--hints-input "$hints_path" --augment-threshold "$AUGMENT_THRESHOLD")
  fi

  echo "[Phase 1] Inference + EM (${variant_label})" | tee "$log_path"
  if ! "${inf_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! ${variant_label} ${tag} Phase 1 failed. See ${log_path}" | tee -a "$log_path"
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
  echo "[Phase 2] Execution accuracy (${variant_label})" | tee -a "$log_path"
  if ! "${ex_cmd[@]}" 2>&1 | tee -a "$log_path"; then
    echo "!! ${variant_label} ${tag} Phase 2 failed. Phase 1 artifacts preserved in ${outdir}" | tee -a "$log_path"
  fi
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  mkdir -p "$BASELINE_ROOT"
  for TAG in $SIZES; do
    MODEL_ID="$(model_id_for "$TAG")"
    if [[ -z "$MODEL_ID" ]]; then
      echo "!! Unknown size tag: $TAG (skipping)"
      continue
    fi
    run_model_variant "baseline" "$BASELINE_ROOT" "0" "$TAG" "$MODEL_ID"
  done
fi

if [[ "$RUN_AUG" == "1" ]]; then
  mkdir -p "$AUG_ROOT"
  for TAG in $SIZES; do
    MODEL_ID="$(model_id_for "$TAG")"
    if [[ -z "$MODEL_ID" ]]; then
      echo "!! Unknown size tag: $TAG (skipping)"
      continue
    fi
    run_model_variant "augmented" "$AUG_ROOT" "1" "$TAG" "$MODEL_ID"
  done
fi

SUMMARY_LOG="${BASELINE_ROOT}/summary.log"
mkdir -p "$(dirname "$SUMMARY_LOG")"

echo
echo "============================================================"
echo " Summary of EM/EX per model"
echo "============================================================"
AUGMENT_THRESHOLD="$AUGMENT_THRESHOLD" python - <<'PY' | tee "$SUMMARY_LOG"
import json
import os
from pathlib import Path


def collect(root: Path):
    rows = {}
    if not root.exists():
        return rows
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        ex_file = sub / "eval_ex.json"
        em_file = sub / "eval_em_only.json"
        if ex_file.exists():
            data = json.loads(ex_file.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            rows[sub.name] = (
                summary.get("count", 0),
                summary.get("exact_match"),
                summary.get("execution_accuracy"),
            )
        elif em_file.exists():
            data = json.loads(em_file.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            rows[sub.name] = (
                summary.get("count", 0),
                summary.get("exact_match"),
                None,
            )
    return rows


def fmt(value):
    return f"{value:.4f}" if value is not None else "  n/a  "


def diff(new, base):
    if new is None or base is None:
        return "  n/a  "
    return f"{new - base:+.4f}"


baseline = collect(Path("zero_shot/results/qwen_compare"))
augmented = collect(Path("zero_shot/results/qwen_compare_aug"))
threshold = os.environ.get("AUGMENT_THRESHOLD", "0.40")

print("=== Baseline ===")
print(f"{'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}")
for name in sorted(baseline):
    n, em, ex = baseline[name]
    print(f"{name:>8}  {n:>5}  {fmt(em):>8}  {fmt(ex):>8}")

if augmented:
    print()
    print(f"=== Augmented (threshold={threshold}) ===")
    print(f"{'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}  {'dEM':>8}  {'dEX':>8}")
    for name in sorted(augmented):
        n, em, ex = augmented[name]
        base_em, base_ex = (None, None)
        if name in baseline:
            _, base_em, base_ex = baseline[name]
        print(
            f"{name:>8}  {n:>5}  {fmt(em):>8}  {fmt(ex):>8}  "
            f"{diff(em, base_em):>8}  {diff(ex, base_ex):>8}"
        )
PY

echo "Summary saved to ${SUMMARY_LOG}"
