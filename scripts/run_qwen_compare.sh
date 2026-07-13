#!/usr/bin/env bash
# Compare Qwen2.5-Coder sizes on ViSpider with two independent phases:
#   Phase 1: inference + EM only, saves predictions/gold/eval_em_only.json
#   Phase 2: execution accuracy from saved artifacts, saves exec_details/eval_ex.json

set -euo pipefail

DATASET="${DATASET:-vispider}"
SPLIT="${SPLIT:-dev}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LIMIT="${LIMIT:-}"
SIZES="${SIZES:-0_5B 1_5B 3B 7B}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"

RESULTS_ROOT="zero_shot/results/qwen_compare"
mkdir -p "$RESULTS_ROOT"

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

for TAG in $SIZES; do
  MODEL_ID="$(model_id_for "$TAG")"
  if [[ -z "$MODEL_ID" ]]; then
    echo "!! Unknown size tag: $TAG (skipping)"
    continue
  fi

  OUTDIR="${RESULTS_ROOT}/${TAG}"
  mkdir -p "$OUTDIR"

  PRED_PATH="${OUTDIR}/predictions.txt"
  GOLD_PATH="${OUTDIR}/gold.txt"
  EM_PATH="${OUTDIR}/eval_em_only.json"
  EX_PATH="${OUTDIR}/eval_ex.json"
  DETAILS_PATH="${OUTDIR}/exec_details.json"
  LOG_PATH="${OUTDIR}/run.log"

  echo "============================================================"
  echo " Running ${TAG}: ${MODEL_ID}"
  echo " -> ${OUTDIR}"
  echo "============================================================"

  INF_CMD=(python -m zero_shot.run_zero_shot
        --mode inference
        --dataset "$DATASET"
        --split "$SPLIT"
        --model "$MODEL_ID"
        --output "$EM_PATH"
        --predictions-output "$PRED_PATH"
        --gold-output "$GOLD_PATH"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL")

  if [[ -n "$LIMIT" ]]; then
    INF_CMD+=(--limit "$LIMIT")
  fi

  echo "[Phase 1] Inference + EM" | tee "$LOG_PATH"
  if ! "${INF_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"; then
    echo "!! ${TAG} Phase 1 failed. See ${LOG_PATH}" | tee -a "$LOG_PATH"
    continue
  fi

  EX_CMD=(python -m zero_shot.run_zero_shot
        --mode exec
        --dataset "$DATASET"
        --split "$SPLIT"
        --model "$MODEL_ID"
        --predictions-input "$PRED_PATH"
        --gold-input "$GOLD_PATH"
        --em-input "$EM_PATH"
        --output "$EX_PATH"
        --exec-details-output "$DETAILS_PATH"
        --timeout-seconds "$TIMEOUT_SECONDS")

  echo "" | tee -a "$LOG_PATH"
  echo "[Phase 2] Execution accuracy" | tee -a "$LOG_PATH"
  if ! "${EX_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"; then
    echo "!! ${TAG} Phase 2 failed. Phase 1 artifacts are preserved in ${OUTDIR}" | tee -a "$LOG_PATH"
  fi
done

SUMMARY_LOG="${RESULTS_ROOT}/summary.log"

echo
echo "============================================================"
echo " Summary of EM/EX per model"
echo "============================================================"
python - <<'PY' | tee "$SUMMARY_LOG"
import json
from pathlib import Path
root = Path("zero_shot/results/qwen_compare")
rows = []
for sub in sorted(root.iterdir()):
    if not sub.is_dir():
        continue
    ex_file = sub / "eval_ex.json"
    em_file = sub / "eval_em_only.json"
    if ex_file.exists():
        data = json.load(open(ex_file, encoding="utf-8"))
        s = data["summary"]
        rows.append((sub.name, s.get("count", 0), s.get("exact_match"), s.get("execution_accuracy")))
    elif em_file.exists():
        data = json.load(open(em_file, encoding="utf-8"))
        s = data["summary"]
        rows.append((sub.name, s.get("count", 0), s.get("exact_match"), None))
print(f"{'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}")
for name, n, em, ex in rows:
    em_s = f"{em:.4f}" if em is not None else "  n/a  "
    ex_s = f"{ex:.4f}" if ex is not None else "  n/a  "
    print(f"{name:>8}  {n:>5}  {em_s:>8}  {ex_s:>8}")
PY

echo "Summary saved to ${SUMMARY_LOG}"
