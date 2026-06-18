"""
Spider evaluation wrapper for computing EM and EX metrics.

Uses the original, unmodified `evaluation.py` from taoyds/spider (vendored
into spider_repo/, with a single `return scores` added at the end of
`evaluate()` so it can be called as a library function instead of only
being run as a CLI script). No scoring logic is changed.
"""

import sys
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# Add spider_repo to path for evaluation import
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
    Run Spider evaluation and return structured metrics.

    Args:
        gold_path: Path to gold SQL file (format: "{sql}\\t{db_id}" per line)
        pred_path: Path to predictions file (one SQL per line)
        db_dir: Directory containing SQLite database files
        table_path: Path to tables.json for foreign key mapping
        etype: Evaluation type, per original Spider script - "match" (EM only),
               "exec" (EX only), or "all" (both EM and EX). Note: the original
               evaluate() only computes 'exact' when etype in ("all", "match"),
               and only computes 'exec' when etype in ("all", "exec") - the two
               are independent, so "exec" alone will NOT compute EM.

    Returns:
        Dict with structure:
        {
            "all": {"count": N, "exact_match": X, "execution_accuracy": Z},
            "easy": {...},
            "medium": {...},
            "hard": {...},
            "extra": {...}
        }
    """
    # Build foreign key map from tables.json
    kmaps = spider_eval.build_foreign_key_map_from_json(table_path)

    # Capture stdout since spider_eval.evaluate() prints directly
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Run evaluation (positional args only)
        scores = spider_eval.evaluate(
            gold_path,
            pred_path,
            db_dir,
            etype,
            kmaps
        )

        # Capture output
        output = sys.stdout.getvalue()

    finally:
        sys.stdout = old_stdout

    # Parse evaluation output
    # spider_eval.evaluate() returns dict with keys like:
    # 'all', 'easy', 'medium', 'hard', 'extra'
    # Each has: 'count', 'exact' (if etype in ['all', 'match']),
    # 'exec' (if etype in ['all', 'exec'])

    result = {}
    for difficulty in ['all', 'easy', 'medium', 'hard', 'extra']:
        if difficulty in scores:
            level_scores = scores[difficulty]
            result[difficulty] = {
                "count": level_scores.get('count', 0),
                "exact_match": level_scores.get('exact', 0.0),
            }
            if etype in ("exec", "all"):
                result[difficulty]["execution_accuracy"] = level_scores.get('exec', 0.0)

    return result


def save_results(
    predictions: List[Dict],
    scores: Dict,
    meta: Dict,
    output_path: str
):
    """
    Save full evaluation results to JSON file.

    Args:
        predictions: List of prediction dicts with fields:
            - id, db_id, question, gold_sql, pred_sql
        scores: Metrics dict from run_evaluation()
        meta: Metadata dict with keys: model, dataset, timestamp
        output_path: Path to output JSON file
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

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def parse_gold_file(gold_path: str) -> List[Tuple[str, str]]:
    """
    Parse gold SQL file.

    Args:
        gold_path: Path to gold file (format: "{sql}\\t{db_id}" per line)

    Returns:
        List of (sql, db_id) tuples
    """
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


def parse_pred_file(pred_path: str) -> List[str]:
    """
    Parse predictions file.

    Args:
        pred_path: Path to predictions file (one SQL per line)

    Returns:
        List of predicted SQL strings
    """
    predictions = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(line.strip())
    return predictions
