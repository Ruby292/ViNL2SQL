"""End-to-end schema linking pipeline using dict_list filtering."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from shared.spider_eval import run_evaluation, save_results
from schema_linking.dict_list.candidate_filter import filter_schema
from schema_linking.dict_list.prompt_builder import build_prompt
from schema_linking.dict_list.qwen_client import batch_call_qwen, load_model


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BASE_DIR / "data"
SPIDER_DB = DATA_ROOT / "spider_db"
VISPIDER_DIR = DATA_ROOT / "vispider_data"
TABLE_FILE = VISPIDER_DIR / "tables.json"
RESULTS_DIR = BASE_DIR / "schema_linking" / "dict_list" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema linking pipeline with dict_list filtering")
    parser.add_argument("--dev", type=str, default=str(VISPIDER_DIR / "vispider_dev.json"))
    parser.add_argument("--dict_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-exec", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dict_list(path: str) -> Dict[str, Dict]:
    records = load_json(path)
    return {item["db_id"]: item for item in records}


def save_lines(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def build_detailed_predictions(examples: List[Dict], gold_data: List[Tuple[str, str]], predictions: List[str]) -> List[Dict]:
    detailed = []
    for idx, (example, (gold_sql, db_id), pred_sql) in enumerate(zip(examples, gold_data, predictions)):
        detailed.append(
            {
                "id": idx,
                "example_id": example.get("id", f"example-{idx}"),
                "db_id": db_id,
                "question": example.get("question_vi", example.get("question", "")),
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
            }
        )
    return detailed


def main() -> None:
    args = parse_args()

    examples = load_json(args.dev)
    dict_list = load_dict_list(args.dict_list)

    if args.limit is not None:
        examples = examples[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_txt_path = output_dir / "predictions.txt"
    gold_txt_path = output_dir / "gold.txt"
    output_json_path = output_dir / "output.json"

    llm = sampling_params = None
    if args.backend == "vllm":
        llm, sampling_params = load_model(args.model, args.max_model_len, args.gpu_memory_utilization)

    predictions: List[str] = []
    gold_data: List[Tuple[str, str]] = []

    prompts = []
    for example in examples:
        question = example.get("question_vi") or example.get("question") or ""
        db_id = example["db_id"]
        gold_sql = example["query"]

        candidate_schema = filter_schema(question, db_id, dict_list)
        prompt = build_prompt(question, candidate_schema, db_id)
        prompts.append(prompt)
        gold_data.append((gold_sql, db_id))

    predictions = batch_call_qwen(prompts, llm=llm, sampling_params=sampling_params, model_name=args.model)

    if len(predictions) != len(gold_data):
        min_len = min(len(predictions), len(gold_data))
        predictions = predictions[:min_len]
        gold_data = gold_data[:min_len]
        examples = examples[:min_len]

    save_lines(predictions, pred_txt_path)
    save_lines([f"{sql}\t{db_id}" for sql, db_id in gold_data], gold_txt_path)

    etype = "match" if args.disable_exec else "all"
    scores = run_evaluation(
        gold_path=str(gold_txt_path),
        pred_path=str(pred_txt_path),
        db_dir=str(SPIDER_DB),
        table_path=str(TABLE_FILE),
        etype=etype,
    )

    detailed_predictions = build_detailed_predictions(examples, gold_data, predictions)
    meta = {
        "model": args.model,
        "dataset": "vispider_dev",
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(examples),
        "evaluation_type": etype,
    }

    save_results(detailed_predictions, scores, meta, str(output_json_path))
    print(f"Saved predictions to {pred_txt_path}")
    print(f"Saved gold to {gold_txt_path}")
    print(f"Saved evaluation to {output_json_path}")


if __name__ == "__main__":
    main()
