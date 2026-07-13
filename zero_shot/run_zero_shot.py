"""
Zero-shot Text-to-SQL pipeline orchestrator.

Split into two independent phases so the inference results are not lost if
execution evaluation crashes or hangs.

Phase 1 (inference, default):
    python -m zero_shot.run_zero_shot --mode inference --split dev \
        --model Qwen/Qwen2.5-Coder-7B-Instruct

    Runs the model, extracts pred_sql, computes EM only (no SQLite execution)
    and writes:
        predictions.txt
        gold.txt
        eval_em_only.json

Phase 2 (execution accuracy):
    python -m zero_shot.run_zero_shot --mode exec \
        --predictions-input .../predictions.txt \
        --gold-input       .../gold.txt

    Reads the persisted artifacts, runs each pred_sql / gold_sql on the matching
    <db_id>.sqlite with a per-query timeout, and writes:
        exec_details.json
        eval_ex.json (merges EM summary from eval_em_only.json when available)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

from zero_shot.prompts import build_prompt, extract_sql
from shared.spider_eval import (
    parse_gold_file,
    parse_pred_file,
    run_exact_match_evaluation,
    run_execution_evaluation,
)


BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
SPIDER_DB = DATA_ROOT / "spider_db"
VISPIDER_DIR = DATA_ROOT / "vispider_data"
RESULTS_DIR = BASE_DIR / "zero_shot" / "results"
TABLE_FILE = VISPIDER_DIR / "tables.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot Text-to-SQL inference and evaluation pipeline"
    )
    parser.add_argument("--mode", choices=["inference", "exec"], default="inference",
                        help="Pipeline phase (default: inference)")
    parser.add_argument("--dataset", default="vispider", choices=["vispider", "vibird"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="HuggingFace model ID (also stored in EM metadata)")
    parser.add_argument("--output", default=None,
                        help="Path to the phase's JSON summary (eval_em_only.json or eval_ex.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples (inference mode only, for smoke tests)")
    parser.add_argument("--predictions-input", default=None,
                        help="Path to existing predictions.txt (implies --mode exec if not set)")
    parser.add_argument("--gold-input", default=None,
                        help="Path to existing gold.txt (execution mode)")
    parser.add_argument("--predictions-output", default=None,
                        help="Where to save predictions.txt in inference mode")
    parser.add_argument("--gold-output", default=None,
                        help="Where to save gold.txt in inference mode")
    parser.add_argument("--exec-details-output", default=None,
                        help="Path to per-example exec_details.json (execution mode)")
    parser.add_argument("--em-input", default=None,
                        help="Optional eval_em_only.json to merge into eval_ex.json")
    parser.add_argument("--timeout-seconds", type=float, default=30.0,
                        help="Per-query SQL timeout in seconds (execution mode)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return parser.parse_args()


def load_dataset(dataset: str, split: str) -> List[Dict]:
    path_map = {
        ("vispider", "dev"): VISPIDER_DIR / "vispider_dev.json",
        ("vispider", "train"): VISPIDER_DIR / "vispider_train.json",
        ("vispider", "test"): VISPIDER_DIR / "vispider_test.json",
    }
    path = path_map.get((dataset, split))
    if path is None:
        raise ValueError(f"Unknown dataset/split combination: {dataset}/{split}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tables(table_path: Path) -> Dict:
    with open(table_path, "r", encoding="utf-8") as f:
        tables_list = json.load(f)
    return {db["db_id"]: db for db in tables_list}


def load_gold(dataset: str, split: str) -> List[Tuple[str, str]]:
    gold_file_map = {
        ("vispider", "dev"): VISPIDER_DIR / "dev_gold.sql",
        ("vispider", "train"): VISPIDER_DIR / "train_gold.sql",
        ("vispider", "test"): VISPIDER_DIR / "test_gold.sql",
    }
    gold_path = gold_file_map.get((dataset, split))
    if gold_path is None or not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    return parse_gold_file(str(gold_path))


def load_model(model_name: str, max_model_len: int, gpu_memory_utilization: float):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for inference. Install it with: pip install vllm\n"
            "Or run with --mode exec to skip inference."
        )
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    return llm, sampling_params


def run_inference_batch(llm, sampling_params, examples: List[Dict], tables: Dict
                        ) -> Tuple[List[str], List[str]]:
    print(f"Building prompts for {len(examples)} examples...")
    tokenizer = llm.get_tokenizer()

    prompts = []
    for example in tqdm(examples, desc="Building prompts"):
        user_prompt = build_prompt(example["question_vi"], example["db_id"], tables)
        messages = [
            {"role": "system", "content": "You are a helpful SQL expert."},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    print("Running batch inference...")
    outputs = llm.generate(prompts, sampling_params)

    predictions, raw_outputs = [], []
    for output in outputs:
        raw_text = output.outputs[0].text
        predictions.append(" ".join(extract_sql(raw_text).split()))
        raw_outputs.append(raw_text)
    return predictions, raw_outputs


def save_predictions_txt(predictions: List[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")


def save_gold_txt(gold_data: List[Tuple[str, str]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sql, db_id in gold_data:
            f.write(f"{sql}\t{db_id}\n")


def _print_em_summary(scores: Dict):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Exact Match)")
    print("=" * 60)
    all_scores = scores["all"]
    print(f"Total examples: {all_scores['count']}")
    print(f"Exact Match (EM): {all_scores['exact_match']:.2%}")
    print("\nBy Difficulty:")
    for level in ["easy", "medium", "hard", "extra"]:
        if level in scores:
            s = scores[level]
            print(f"  {level.capitalize():8s} ({s['count']:3d}): EM={s['exact_match']:.2%}")
    print("=" * 60)


def _print_ex_summary(ex_scores: Dict, em_summary: Dict = None):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Execution Accuracy)")
    print("=" * 60)
    all_scores = ex_scores["all"]
    print(f"Total examples: {all_scores['count']}")
    print(f"Execution Accuracy (EX): {all_scores['execution_accuracy']:.2%}")
    if em_summary and em_summary.get("exact_match") is not None:
        print(f"Exact Match (EM, from Phase 1): {em_summary['exact_match']:.2%}")
    print("=" * 60)


def run_inference_mode(args) -> int:
    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Model: {args.model}")

    examples = load_dataset(args.dataset, args.split)
    tables = load_tables(TABLE_FILE)
    gold_data = load_gold(args.dataset, args.split)

    if args.limit:
        print(f"Limiting to first {args.limit} examples (smoke test)")
        examples = examples[:args.limit]
        gold_data = gold_data[:args.limit]

    if len(examples) != len(gold_data):
        raise ValueError(
            f"Example count ({len(examples)}) != gold count ({len(gold_data)})"
        )

    tag = f"{args.dataset}_{args.split}"
    pred_txt_path = Path(args.predictions_output) if args.predictions_output \
        else RESULTS_DIR / f"{tag}_predictions.txt"
    gold_txt_path = Path(args.gold_output) if args.gold_output \
        else RESULTS_DIR / f"{tag}_gold.txt"
    output_path = Path(args.output) if args.output \
        else RESULTS_DIR / f"{tag}_eval_em_only.json"

    save_gold_txt(gold_data, gold_txt_path)

    print("\n[Phase 1] Running inference...")
    llm, sampling_params = load_model(
        args.model, args.max_model_len, args.gpu_memory_utilization
    )
    predictions, raw_outputs = run_inference_batch(llm, sampling_params, examples, tables)
    save_predictions_txt(predictions, pred_txt_path)
    print(f"Predictions saved to {pred_txt_path}")
    print(f"Gold saved to {gold_txt_path}")

    print("\n[Phase 1] Computing Exact Match...")
    scores, em_details = run_exact_match_evaluation(
        gold_data=gold_data,
        predictions=predictions,
        db_dir=str(SPIDER_DB),
        table_path=str(TABLE_FILE),
    )

    per_example = []
    for idx, (example, (gold_sql, db_id), pred_sql) in enumerate(
            zip(examples, gold_data, predictions)):
        detail = em_details[idx]
        item = {
            "id": idx,
            "example_id": example.get("id", f"example-{idx}"),
            "db_id": db_id,
            "question": example.get("question_vi", example.get("question", "")),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "raw_output": raw_outputs[idx],
            "hardness": detail["hardness"],
            "exact_match": detail["exact_match"],
        }
        if detail.get("error"):
            item["parse_error"] = detail["error"]
        per_example.append(item)

    output_data = {
        "summary": {
            "count": scores["all"]["count"],
            "exact_match": scores["all"]["exact_match"],
            "model": args.model,
            "dataset": tag,
            "timestamp": datetime.now().isoformat(),
        },
        "by_difficulty": {
            level: {
                "count": scores[level]["count"],
                "exact_match": scores[level]["exact_match"],
            }
            for level in ["easy", "medium", "hard", "extra"]
            if level in scores
        },
        "examples": per_example,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"EM results saved to {output_path}")

    _print_em_summary(scores)
    print("\nPhase 1 completed successfully.")
    return 0


def _resolve_exec_paths(args):
    if not args.predictions_input:
        raise ValueError("--predictions-input is required in --mode exec")
    pred_path = Path(args.predictions_input)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    if args.gold_input:
        gold_path = Path(args.gold_input)
    else:
        gold_path = pred_path.with_name("gold.txt")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    if args.output:
        eval_path = Path(args.output)
    else:
        eval_path = pred_path.with_name("eval_ex.json")

    if args.exec_details_output:
        details_path = Path(args.exec_details_output)
    else:
        details_path = pred_path.with_name("exec_details.json")

    if args.em_input:
        em_path = Path(args.em_input)
    else:
        candidate = pred_path.with_name("eval_em_only.json")
        em_path = candidate if candidate.exists() else None
    return pred_path, gold_path, eval_path, details_path, em_path


def run_exec_mode(args) -> int:
    pred_path, gold_path, eval_path, details_path, em_path = _resolve_exec_paths(args)
    print(f"Predictions: {pred_path}")
    print(f"Gold:        {gold_path}")
    print(f"Timeout/query: {args.timeout_seconds}s")

    predictions = parse_pred_file(str(pred_path))
    gold_data = parse_gold_file(str(gold_path))
    if len(predictions) != len(gold_data):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != gold count ({len(gold_data)})"
        )

    print(f"\n[Phase 2] Running execution accuracy on {len(predictions)} examples...")
    ex_scores, exec_details = run_execution_evaluation(
        gold_data=gold_data,
        predictions=predictions,
        db_dir=str(SPIDER_DB),
        table_path=str(TABLE_FILE),
        timeout_seconds=args.timeout_seconds,
    )

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(exec_details, f, indent=2, ensure_ascii=False)
    print(f"Exec details saved to {details_path}")

    em_summary = None
    if em_path is not None and em_path.exists():
        try:
            with open(em_path, "r", encoding="utf-8") as f:
                em_data = json.load(f)
            em_summary = em_data.get("summary")
            print(f"Merged EM summary from {em_path}")
        except Exception as exc:
            print(f"WARN: failed to read EM summary from {em_path}: {exc}")

    summary = {
        "count": ex_scores["all"]["count"],
        "execution_accuracy": ex_scores["all"]["execution_accuracy"],
        "model": em_summary.get("model") if em_summary else args.model,
        "dataset": em_summary.get("dataset") if em_summary else f"{args.dataset}_{args.split}",
        "timestamp": datetime.now().isoformat(),
        "timeout_seconds": args.timeout_seconds,
    }
    if em_summary and em_summary.get("exact_match") is not None:
        summary["exact_match"] = em_summary["exact_match"]

    total = ex_scores["all"]["count"]
    timeouts = sum(1 for d in exec_details if d["timeout"])
    errors = sum(1 for d in exec_details if d["error"])
    matches = sum(1 for d in exec_details if d["exec_match"])
    summary["exec_stats"] = {
        "matches": matches,
        "errors": errors,
        "timeouts": timeouts,
        "total": total,
    }

    output_data = {"summary": summary}
    if em_summary is not None:
        output_data["em_summary"] = em_summary

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"EX summary saved to {eval_path}")

    _print_ex_summary(ex_scores, em_summary)
    print(f"Errors: {errors}, Timeouts: {timeouts}, Matches: {matches}/{total}")
    print("Phase 2 completed successfully.")
    return 0


def main():
    args = parse_args()

    if args.predictions_input and args.mode == "inference":
        args.mode = "exec"

    print("=" * 60)
    print(f"Zero-shot Text-to-SQL Pipeline - Mode: {args.mode}")
    print("=" * 60)

    if args.mode == "inference":
        return run_inference_mode(args)
    return run_exec_mode(args)


if __name__ == "__main__":
    sys.exit(main())
