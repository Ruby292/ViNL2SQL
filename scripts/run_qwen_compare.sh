#!/usr/bin/env bash
# Compare 4 Qwen2.5-Coder sizes on ViSpider dev split.
# Meant to be run on a Vast.ai GPU instance (Linux + CUDA + vLLM installed).
#
# Each model gets its own folder:
#   zero_shot/results/qwen_compare/<TAG>/
#     eval.json         # summary EM/EX + per-example (question, gold, pred, raw_output)
#     predictions.txt   # one SQL per line (after extract_sql + collapse newlines)
#     gold.txt          # "{sql}\t{db_id}" per line
#     run.log           # full stdout/stderr of that run
#
# Usage:
#   bash scripts/run_qwen_compare.sh                    # 4 sizes: 0.5B, 1.5B, 3B, 7B
#   LIMIT=20 bash scripts/run_qwen_compare.sh           # smoke test on 20 samples
#   SIZES="1_5B 7B" bash scripts/run_qwen_compare.sh    # only run selected sizes
#   SPLIT=test bash scripts/run_qwen_compare.sh         # different split

set -euo pipefail

# --- config ---
DATASET="${DATASET:-vispider}"
SPLIT="${SPLIT:-dev}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LIMIT="${LIMIT:-}"
SIZES="${SIZES:-0_5B 1_5B 3B 7B}"

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

  echo "============================================================"
  echo " Running ${TAG}: ${MODEL_ID}"
  echo " -> ${OUTDIR}"
  echo "============================================================"

  CMD=(python -m zero_shot.run_zero_shot
        --dataset "$DATASET"
        --split "$SPLIT"
        --model "$MODEL_ID"
        --output "${OUTDIR}/eval.json"
        --predictions-output "${OUTDIR}/predictions.txt"
        --gold-output "${OUTDIR}/gold.txt"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL")

  if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
  fi

  # Run and tee log; keep going to next model if one fails.
  if ! "${CMD[@]}" 2>&1 | tee "${OUTDIR}/run.log"; then
    echo "!! ${TAG} failed. See ${OUTDIR}/run.log"
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
    ev = sub / "eval.json"
    if not ev.exists():
        continue
    data = json.load(open(ev, encoding="utf-8"))
    s = data["summary"]
    rows.append((sub.name, s["count"], s["exact_match"], s.get("execution_accuracy")))
print(f"{'model':>8}  {'N':>5}  {'EM':>8}  {'EX':>8}")
for name, n, em, ex in rows:
    ex_s = f"{ex:.4f}" if ex is not None else "  n/a  "
    print(f"{name:>8}  {n:>5}  {em:.4f}  {ex_s}")
PY

echo "Summary saved to ${SUMMARY_LOG}"
