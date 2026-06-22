"""
Shared Spider evaluation wrapper.

This module keeps Spider scoring logic vendored in `spider_repo/` and only
provides a thin adapter for project pipelines.
"""

import sys
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "spider_repo"))

import evaluation as spider_eval


def run_evaluation(
    gold_path: str,
    pred_path: str,
    db_dir: str,
    table_path: str,
    etype: str = "match"
) -> Dict:
    """
    Run Spider evaluation and return normalized metrics.

    Args:
        gold_path: Path to gold SQL file (format: "{sql}\t{db_id}" per line)
        pred_path: Path to predictions file (one SQL per line)
        db_dir: Directory containing SQLite database files
        table_path: Path to tables.json for foreign key mapping
        etype: Evaluation type, per original Spider script - "match", "exec", or "all"

    Returns:
        Dict keyed by difficulty with normalized score names.
    """
    kmaps = spider_eval.build_foreign_key_map_from_json(table_path)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        scores = spider_eval.evaluate(
            gold_path,
            pred_path,
            db_dir,
            etype,
            kmaps
        )
        _ = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    result = {}
    for difficulty in ["all", "easy", "medium", "hard", "extra"]:
        if difficulty in scores:
            level_scores = scores[difficulty]
            result[difficulty] = {
                "count": level_scores.get("count", 0),
                "exact_match": level_scores.get("exact", 0.0),
            }
            if etype in ("exec", "all"):
                result[difficulty]["execution_accuracy"] = level_scores.get("exec", 0.0)

    return result


def save_results(
    predictions: List[Dict],
    scores: Dict,
    meta: Dict,
    output_path: str
):
    """
    Save evaluation results to JSON.
    """
    output_data = {
        "summary": {
            "count": scores["all"]["count"],
            "exact_match": scores["all"]["exact_match"],
            "execution_accuracy": scores["all"].get("execution_accuracy"),
            "model": meta.get("model", "unknown"),
            "dataset": meta.get("dataset", "unknown"),
            "timestamp": meta.get("timestamp", datetime.now().isoformat())
        },
        "by_difficulty": {
            level: {
                "count": scores[level]["count"],
                "exact_match": scores[level]["exact_match"],
                "execution_accuracy": scores[level].get("execution_accuracy")
            }
            for level in ["easy", "medium", "hard", "extra"]
            if level in scores
        },
        "predictions": predictions
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def parse_gold_file(gold_path: str) -> List[Tuple[str, str]]:
    """
    Parse gold SQL file.
    """
    gold_data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                sql, db_id = parts[0], parts[1]
                gold_data.append((sql, db_id))
    return gold_data


def parse_pred_file(pred_path: str) -> List[str]:
    """
    Parse predictions file.
    """
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            predictions.append(line.strip())
    return predictions
