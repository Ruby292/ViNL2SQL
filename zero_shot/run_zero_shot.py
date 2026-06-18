"""
Zero-shot Text-to-SQL pipeline orchestrator.

Usage:
    python -m zero_shot.run_zero_shot --dataset vispider --split dev --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, TYPE_CHECKING
from tqdm import tqdm

# Conditional import - vLLM only needed for inference phase
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

from zero_shot.prompts import build_prompt, extract_sql
from zero_shot.spider_eval import run_evaluation, save_results


# Path configuration
BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
SPIDER_DB = DATA_ROOT / "spider_db"
VISPIDER_DIR = DATA_ROOT / "vispider_data"
RESULTS_DIR = BASE_DIR / "zero_shot" / "results"
TABLE_FILE = VISPIDER_DIR / "tables.json"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Zero-shot Text-to-SQL inference and evaluation pipeline"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="vispider",
        choices=["vispider", "vibird"],
        help="Dataset to use (default: vispider)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test", "train"],
        help="Dataset split (default: dev)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID for inference"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: results/{dataset}_{split}_zeroshot.json)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for smoke testing)"
    )

    parser.add_argument(
        "--disable-exec",
        action="store_true",
        help="Disable execution accuracy evaluation (only compute EM)"
    )

    parser.add_argument(
        "--predictions-input",
        type=str,
        default=None,
        help="Path to existing predictions.txt file (skip inference phase)"
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (default: 0.7)"
    )

    return parser.parse_args()


def load_dataset(dataset: str, split: str) -> List[Dict]:
    """
    Load dataset examples from JSON file.

    Args:
        dataset: Dataset name (e.g., "vispider")
        split: Dataset split (e.g., "dev", "test", "train")

    Returns:
        List of example dicts with keys: id, db_id, question_vi, question, query
    """
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

    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    return examples


def load_tables(table_path: Path) -> Dict:
    """
    Load tables.json and convert to dict keyed by db_id.

    Args:
        table_path: Path to tables.json

    Returns:
        Dict mapping db_id to table schema
    """
    with open(table_path, 'r', encoding='utf-8') as f:
        tables_list = json.load(f)

    # Convert list to dict keyed by db_id
    tables = {db['db_id']: db for db in tables_list}
    return tables


def load_gold(dataset: str, split: str) -> List[Tuple[str, str]]:
    """
    Load gold SQL file.

    Args:
        dataset: Dataset name
        split: Dataset split

    Returns:
        List of (sql, db_id) tuples
    """
    gold_file_map = {
        ("vispider", "dev"): VISPIDER_DIR / "dev_gold.sql",
        ("vispider", "train"): VISPIDER_DIR / "train_gold.sql",
        ("vispider", "test"): VISPIDER_DIR / "test_gold.sql",
    }

    gold_path = gold_file_map.get((dataset, split))
    if gold_path is None or not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold_data = []
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                sql, db_id = parts[0], parts[1]
                gold_data.append((sql, db_id))

    return gold_data


def load_model(model_name: str, max_model_len: int, gpu_memory_utilization: float):
    """
    Load vLLM model for inference.

    Args:
        model_name: HuggingFace model ID
        max_model_len: Maximum context length
        gpu_memory_utilization: GPU memory utilization fraction

    Returns:
        Tuple of (LLM, SamplingParams)
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for inference. Install it with: pip install vllm\n"
            "Or use --predictions-input to run evaluation-only mode without vLLM."
        )

    print(f"Loading model: {model_name}")

    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    return llm, sampling_params


def run_inference_batch(
    llm,
    sampling_params,
    examples: List[Dict],
    tables: Dict
) -> List[str]:
    """
    Run batch inference on all examples.

    Args:
        llm: vLLM model instance
        sampling_params: Sampling parameters
        examples: List of example dicts
        tables: Tables schema dict

    Returns:
        List of predicted SQL strings
    """
    print(f"Building prompts for {len(examples)} examples...")

    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()

    # Build all prompts with chat template
    prompts = []
    for example in tqdm(examples, desc="Building prompts"):
        user_prompt = build_prompt(example['question_vi'], example['db_id'], tables)

        # Apply chat template
        messages = [
            {"role": "system", "content": "You are a helpful SQL expert."},
            {"role": "user", "content": user_prompt}
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(formatted_prompt)

    print(f"Running batch inference...")

    # Generate batch
    outputs = llm.generate(prompts, sampling_params)

    # Extract SQL from each output
    predictions = []
    for output in outputs:
        raw_text = output.outputs[0].text
        sql = extract_sql(raw_text)
        predictions.append(sql)

    return predictions


def save_predictions_txt(predictions: List[str], output_path: Path):
    """Save predictions to text file (one SQL per line)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')


def save_gold_txt(gold_data: List[Tuple[str, str]], output_path: Path):
    """Save gold data to text file (format: {sql}\\t{db_id})"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sql, db_id in gold_data:
            f.write(f"{sql}\t{db_id}\n")


def load_predictions_txt(pred_path: str) -> List[str]:
    """Load predictions from text file"""
    with open(pred_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def build_detailed_predictions(
    examples: List[Dict],
    gold_data: List[Tuple[str, str]],
    predictions: List[str],
    scores: Dict
) -> List[Dict]:
    """
    Build detailed prediction list with per-example metrics.

    Args:
        examples: List of example dicts
        gold_data: List of (gold_sql, db_id) tuples
        predictions: List of predicted SQL strings
        scores: Evaluation scores dict

    Returns:
        List of detailed prediction dicts
    """
    detailed = []

    for idx, (example, (gold_sql, db_id), pred_sql) in enumerate(zip(examples, gold_data, predictions)):
        detailed.append({
            "id": idx,
            "example_id": example.get('id', f'example-{idx}'),
            "db_id": db_id,
            "question": example.get('question_vi', example.get('question', '')),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
        })

    return detailed


def print_summary(scores: Dict):
    """Print evaluation summary"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    all_scores = scores['all']
    print(f"Total examples: {all_scores['count']}")
    print(f"Exact Match (EM):         {all_scores['exact_match']:.2%}")

    if 'execution_accuracy' in all_scores and all_scores['execution_accuracy'] is not None:
        print(f"Execution Accuracy (EX):   {all_scores['execution_accuracy']:.2%}")

    print("\nBy Difficulty:")
    for level in ['easy', 'medium', 'hard', 'extra']:
        if level in scores:
            level_scores = scores[level]
            ex_str = (f", EX={level_scores['execution_accuracy']:.2%}"
                      if level_scores.get('execution_accuracy') is not None else "")
            print(f"  {level.capitalize():8s} ({level_scores['count']:3d}): "
                  f"EM={level_scores['exact_match']:.2%}{ex_str}")

    print("=" * 60)


def main():
    """Main pipeline orchestrator"""
    args = parse_args()

    # Set default output path
    if args.output is None:
        args.output = str(RESULTS_DIR / f"{args.dataset}_{args.split}_zeroshot.json")

    print("=" * 60)
    print("Zero-shot Text-to-SQL Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # ── Phase 1: Load data ──
    print("\n[Phase 1] Loading data...")
    examples = load_dataset(args.dataset, args.split)
    tables = load_tables(TABLE_FILE)
    gold_data = load_gold(args.dataset, args.split)

    if args.limit:
        print(f"Limiting to first {args.limit} examples (smoke test)")
        examples = examples[:args.limit]
        gold_data = gold_data[:args.limit]

    print(f"Loaded {len(examples)} examples")

    # Prepare intermediate files
    pred_txt_path = RESULTS_DIR / f"{args.dataset}_{args.split}_predictions.txt"
    gold_txt_path = RESULTS_DIR / f"{args.dataset}_{args.split}_gold.txt"

    # Save gold file for evaluation
    save_gold_txt(gold_data, gold_txt_path)

    # ── Phase 2: Inference (skip if predictions-input provided) ──
    if args.predictions_input:
        print(f"\n[Phase 2] Loading existing predictions from {args.predictions_input}...")
        predictions = load_predictions_txt(args.predictions_input)

        # Validate and adjust predictions length to match examples
        if len(predictions) < len(examples):
            raise ValueError(
                f"Predictions file has {len(predictions)} entries but dataset has {len(examples)}. "
                f"Cannot evaluate with missing predictions. "
                f"Use --limit {len(predictions)} to match the predictions file."
            )
        elif len(predictions) > len(examples):
            print(f"Truncating predictions from {len(predictions)} to {len(examples)} to match dataset")
            predictions = predictions[:len(examples)]

        # Save predictions to canonical location for evaluation
        save_predictions_txt(predictions, pred_txt_path)
        print(f"Predictions saved to {pred_txt_path}")
    else:
        print("\n[Phase 2] Running inference...")
        llm, sampling_params = load_model(
            args.model,
            args.max_model_len,
            args.gpu_memory_utilization
        )

        predictions = run_inference_batch(llm, sampling_params, examples, tables)

        # Save predictions
        save_predictions_txt(predictions, pred_txt_path)
        print(f"Predictions saved to {pred_txt_path}")

    # ── Phase 3: Evaluation ──
    print("\n[Phase 3] Running evaluation...")
    # NOTE: in the original Spider evaluate(), 'exact' (EM) is only computed
    # when etype in ("all", "match"), and 'exec' (EX) only when etype in
    # ("all", "exec") - they are independent branches, not nested. Using
    # etype="exec" alone would silently skip EM entirely. To get both EM and
    # EX we must use "all".
    etype = "match" if args.disable_exec else "all"

    # Validate predictions and gold data have same length
    if len(predictions) != len(gold_data):
        print(f"ERROR: Prediction count ({len(predictions)}) != gold count ({len(gold_data)})")
        print("Truncating to minimum length for evaluation")
        min_len = min(len(predictions), len(gold_data))
        predictions = predictions[:min_len]
        gold_data = gold_data[:min_len]
        examples = examples[:min_len]
        # Re-save truncated files
        save_predictions_txt(predictions, pred_txt_path)
        save_gold_txt(gold_data, gold_txt_path)

    scores = run_evaluation(
        gold_path=str(gold_txt_path),
        pred_path=str(pred_txt_path),
        db_dir=str(SPIDER_DB),
        table_path=str(TABLE_FILE),
        etype=etype
    )

    # ── Phase 4: Save full results ──
    print("\n[Phase 4] Saving results...")

    detailed_predictions = build_detailed_predictions(examples, gold_data, predictions, scores)

    meta = {
        "model": args.model,
        "dataset": f"{args.dataset}_{args.split}",
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(examples),
        "evaluation_type": etype,
    }

    save_results(detailed_predictions, scores, meta, args.output)
    print(f"Full results saved to {args.output}")

    # Print summary
    print_summary(scores)

    print("\nPipeline completed successfully!")


if __name__ == '__main__':
    main()
