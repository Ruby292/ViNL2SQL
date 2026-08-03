import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from augmentation.similarity import DEFAULT_EMBEDDING_MODEL
from shared.spider_eval import (
    parse_gold_file,
    parse_pred_file,
    run_exact_match_evaluation,
    run_execution_evaluation,
)
from zero_shot.prompts import build_prompt, build_prompt_augmented, extract_sql


BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
SPIDER_DB = DATA_ROOT / "spider_db"
VISPIDER_DIR = DATA_ROOT / "vispider_data"
RESULTS_DIR = BASE_DIR / "zero_shot" / "results"
TABLE_FILE = VISPIDER_DIR / "tables.json"

DATA_FILES = {
    ("vispider", "dev"): VISPIDER_DIR / "vispider_dev.json",
    ("vispider", "train"): VISPIDER_DIR / "vispider_train.json",
    ("vispider", "test"): VISPIDER_DIR / "vispider_test.json",
}
GOLD_FILES = {
    ("vispider", "dev"): VISPIDER_DIR / "dev_gold.sql",
    ("vispider", "train"): VISPIDER_DIR / "train_gold.sql",
    ("vispider", "test"): VISPIDER_DIR / "test_gold.sql",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot Text-to-SQL inference and evaluation pipeline"
    )
    parser.add_argument("--mode", choices=["inference", "exec"], default="inference")
    parser.add_argument("--dataset", default="vispider", choices=["vispider"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-input")
    parser.add_argument("--gold-input")
    parser.add_argument("--predictions-output")
    parser.add_argument("--gold-output")
    parser.add_argument("--exec-details-output")
    parser.add_argument("--em-input")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--hints-input", type=str, default=None, help="Path to hints.json from augmentation pipeline")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-threshold", type=float, default=0.4)
    parser.add_argument("--augment-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--augment-stats-output")
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset: str, split: str) -> List[Dict]:
    path = DATA_FILES.get((dataset, split))
    if path is None:
        raise ValueError(f"Unknown dataset/split combination: {dataset}/{split}")
    return load_json(path)


def load_gold(dataset: str, split: str) -> List[Tuple[str, str]]:
    path = GOLD_FILES.get((dataset, split))
    if path is None or not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")
    return parse_gold_file(str(path))


def load_tables() -> Dict:
    return {db["db_id"]: db for db in load_json(TABLE_FILE)}


def load_model(model_name: str, max_model_len: int, gpu_memory_utilization: float):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for inference. Install it with: pip install vllm\n"
            "Or run with --mode exec to skip inference."
        )

    print(f"Loading model: {model_name}")
    return (
        LLM(
            model=model_name,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        ),
        SamplingParams(temperature=0.0, max_tokens=512),
    )


def run_inference_batch(
    llm,
    sampling_params,
    examples: List[Dict],
    tables: Dict,
    hints_per_example: List[List[Dict]] = None,
):
    print(f"Building prompts for {len(examples)} examples...")
    tokenizer = llm.get_tokenizer()
    prompts = []

    for idx, example in enumerate(tqdm(examples, desc="Building prompts")):
        if hints_per_example is not None:
            user_prompt = build_prompt_augmented(
                example["question_vi"], example["db_id"], tables, hints_per_example[idx]
            )
        else:
            user_prompt = build_prompt(example["question_vi"], example["db_id"], tables)

        messages = [
            {"role": "system", "content": "You are a helpful SQL expert."},
            {"role": "user", "content": user_prompt},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    print("Running batch inference...")
    predictions, raw_outputs = [], []
    for output in llm.generate(prompts, sampling_params):
        raw_text = output.outputs[0].text
        predictions.append(" ".join(extract_sql(raw_text).split()))
        raw_outputs.append(raw_text)
    return predictions, raw_outputs


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def default_augment_stats_path(output_path: Path) -> Path:
    return output_path.with_name("augment_stats.json")


def print_scores(title: str, metric_name: str, scores: Dict, em_summary: Dict = None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    all_scores = scores["all"]
    print(f"Total examples: {all_scores['count']}")
    print(f"{metric_name}: {next(v for k, v in all_scores.items() if k != 'count'):.2%}")
    if em_summary and em_summary.get("exact_match") is not None:
        print(f"Exact Match (EM, from Phase 1): {em_summary['exact_match']:.2%}")
    print("=" * 60)


def run_inference_mode(args) -> int:
    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Model: {args.model}")

    examples = load_dataset(args.dataset, args.split)
    gold_data = load_gold(args.dataset, args.split)
    if args.limit:
        print(f"Limiting to first {args.limit} examples (smoke test)")
        examples = examples[:args.limit]
        gold_data = gold_data[:args.limit]
    if len(examples) != len(gold_data):
        raise ValueError(f"Example count ({len(examples)}) != gold count ({len(gold_data)})")

    tag = f"{args.dataset}_{args.split}"
    pred_path = Path(args.predictions_output) if args.predictions_output else RESULTS_DIR / f"{tag}_predictions.txt"
    gold_path = Path(args.gold_output) if args.gold_output else RESULTS_DIR / f"{tag}_gold.txt"
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{tag}_eval_em_only.json"
    augment_stats_path = (
        Path(args.augment_stats_output)
        if args.augment_stats_output
        else default_augment_stats_path(output_path)
    )

    write_lines(gold_path, [f"{sql}\t{db_id}" for sql, db_id in gold_data])
    tables = load_tables()
    hints_per_example = None

    if args.hints_input:
        if args.augment:
            raise ValueError("--hints-input and --augment are mutually exclusive")
        with open(args.hints_input, "r", encoding="utf-8") as f:
            hints_data = json.load(f)
        hints_map = {item["index"]: item["hints"] for item in hints_data}
        missing_hint_indexes = [
            i for i in range(len(examples))
            if i not in hints_map
        ]
        if missing_hint_indexes:
            preview = ", ".join(str(i) for i in missing_hint_indexes[:10])
            raise ValueError(
                f"Hints file {args.hints_input} is missing "
                f"{len(missing_hint_indexes)} example indexes. "
                f"First missing indexes: {preview}"
            )
        invalid_hints = [
            (item["index"], hint)
            for item in hints_data
            for hint in item.get("hints", [])
            if "table" not in hint
        ]
        if invalid_hints:
            bad_index, bad_hint = invalid_hints[0]
            raise ValueError(
                f"Hints file {args.hints_input} uses an unsupported hint format "
                f"at index {bad_index}: {bad_hint}. Regenerate hints with the "
                "table-level augmentation pipeline."
            )
        hints_per_example = [hints_map.get(i, []) for i in range(len(examples))]
        print(f"Loaded hints for {len(hints_map)} examples from {args.hints_input}")

    if args.augment:
        from augmentation.pipeline import augment_examples
        from augmentation.similarity import cleanup_encoder

        print(
            f"\n[Phase 1] Running augmentation "
            f"(threshold={args.augment_threshold})..."
        )
        hints_per_example, augment_stats = augment_examples(
            examples=examples,
            tables=tables,
            model_name=args.augment_model,
            threshold=args.augment_threshold,
        )
        write_json(augment_stats_path, augment_stats)
        examples_with_hints = augment_stats["counts"]["examples_with_hints"]
        print(
            f"[Augmentation] {examples_with_hints}/{len(examples)} examples "
            f"have at least one hint."
        )
        print(f"[Augmentation] Stats saved to {augment_stats_path}")
        cleanup_encoder(args.augment_model)

    print("\n[Phase 1] Running inference...")
    llm, sampling_params = load_model(
        args.model, args.max_model_len, args.gpu_memory_utilization
    )
    predictions, raw_outputs = run_inference_batch(
        llm, sampling_params, examples, tables, hints_per_example
    )
    write_lines(pred_path, predictions)
    print(f"Predictions saved to {pred_path}")
    print(f"Gold saved to {gold_path}")

    print("\n[Phase 1] Computing Exact Match...")
    scores, em_details = run_exact_match_evaluation(
        gold_data=gold_data,
        predictions=predictions,
        db_dir=str(SPIDER_DB),
        table_path=str(TABLE_FILE),
    )

    per_example = []
    for idx, (example, (gold_sql, db_id), pred_sql, raw_output, detail) in enumerate(
        zip(examples, gold_data, predictions, raw_outputs, em_details)
    ):
        item = {
            "id": idx,
            "example_id": example.get("id", f"example-{idx}"),
            "db_id": db_id,
            "question": example.get("question_vi", example.get("question", "")),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "raw_output": raw_output,
            "hardness": detail["hardness"],
            "exact_match": detail["exact_match"],
            "hints": hints_per_example[idx] if hints_per_example is not None else [],
        }
        if detail.get("error"):
            item["parse_error"] = detail["error"]
        per_example.append(item)

    augmentation_mode = "none"
    if args.hints_input:
        augmentation_mode = "precomputed_hints"
    elif args.augment:
        augmentation_mode = "on_the_fly"

    write_json(
        output_path,
        {
            "summary": {
                "count": scores["all"]["count"],
                "exact_match": scores["all"]["exact_match"],
                "model": args.model,
                "dataset": tag,
                "timestamp": datetime.now().isoformat(),
                "augmentation": {
                    "mode": augmentation_mode,
                    "hints_input": args.hints_input,
                    "augment_threshold": (
                        args.augment_threshold
                        if args.augment or args.hints_input
                        else None
                    ),
                    "examples_with_hints": (
                        sum(1 for hints in hints_per_example if hints)
                        if hints_per_example is not None
                        else 0
                    ),
                },
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
        },
    )
    print(f"EM results saved to {output_path}")
    print_scores("EVALUATION RESULTS (Exact Match)", "Exact Match (EM)", scores)
    print("\nPhase 1 completed successfully.")
    return 0


def run_exec_mode(args) -> int:
    if not args.predictions_input:
        raise ValueError("--predictions-input is required in --mode exec")

    pred_path = Path(args.predictions_input)
    gold_path = Path(args.gold_input) if args.gold_input else pred_path.with_name("gold.txt")
    eval_path = Path(args.output) if args.output else pred_path.with_name("eval_ex.json")
    details_path = Path(args.exec_details_output) if args.exec_details_output else pred_path.with_name("exec_details.json")
    em_path = Path(args.em_input) if args.em_input else pred_path.with_name("eval_em_only.json")

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

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

    write_json(details_path, exec_details)
    print(f"Exec details saved to {details_path}")

    em_summary = None
    if em_path.exists():
        try:
            em_summary = load_json(em_path).get("summary")
            print(f"Merged EM summary from {em_path}")
        except Exception as exc:
            print(f"WARN: failed to read EM summary from {em_path}: {exc}")

    total = ex_scores["all"]["count"]
    timeouts = sum(1 for detail in exec_details if detail["timeout"])
    errors = sum(1 for detail in exec_details if detail["error"])
    matches = sum(1 for detail in exec_details if detail["exec_match"])
    summary = {
        "count": total,
        "execution_accuracy": ex_scores["all"]["execution_accuracy"],
        "model": em_summary.get("model") if em_summary else args.model,
        "dataset": em_summary.get("dataset") if em_summary else f"{args.dataset}_{args.split}",
        "timestamp": datetime.now().isoformat(),
        "timeout_seconds": args.timeout_seconds,
        "exec_stats": {
            "matches": matches,
            "errors": errors,
            "timeouts": timeouts,
            "total": total,
        },
    }
    if em_summary and em_summary.get("exact_match") is not None:
        summary["exact_match"] = em_summary["exact_match"]
    if em_summary and em_summary.get("augmentation"):
        summary["augmentation"] = em_summary["augmentation"]

    output_data = {"summary": summary}
    if em_summary is not None:
        output_data["em_summary"] = em_summary
    write_json(eval_path, output_data)

    print(f"EX summary saved to {eval_path}")
    print_scores("EVALUATION RESULTS (Execution Accuracy)", "Execution Accuracy (EX)", ex_scores, em_summary)
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

    return run_inference_mode(args) if args.mode == "inference" else run_exec_mode(args)


if __name__ == "__main__":
    sys.exit(main())
