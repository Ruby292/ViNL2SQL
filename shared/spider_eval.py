"""
Shared Spider evaluation wrapper.

Provides:
- run_evaluation: legacy aggregate wrapper around the vendored Spider evaluator.
- run_exact_match_evaluation: EM-only evaluation with per-example details, no SQL execution.
- run_execution_evaluation: EX evaluation with per-query timeout and per-example details.
"""

import sys
import json
import io
import time
import sqlite3
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "spider_repo"))

import evaluation as spider_eval
from process_sql import Schema, get_schema, get_sql


LEVELS = ["easy", "medium", "hard", "extra", "all"]


EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


def run_evaluation(
    gold_path: str,
    pred_path: str,
    db_dir: str,
    table_path: str,
    etype: str = "match"
) -> Dict:
    """Legacy aggregate evaluation wrapper (kept for backward compatibility)."""
    kmaps = spider_eval.build_foreign_key_map_from_json(table_path)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        scores = spider_eval.evaluate(gold_path, pred_path, db_dir, etype, kmaps)
    finally:
        sys.stdout = old_stdout

    result = {}
    for difficulty in LEVELS:
        if difficulty in scores:
            level_scores = scores[difficulty]
            result[difficulty] = {
                "count": level_scores.get("count", 0),
                "exact_match": level_scores.get("exact", 0.0),
            }
            if etype in ("exec", "all"):
                result[difficulty]["execution_accuracy"] = level_scores.get("exec", 0.0)
    return result


def _init_exact_scores() -> Dict:
    return {level: {"count": 0, "exact_match": 0.0} for level in LEVELS}


def _finalize_exact_scores(scores: Dict) -> Dict:
    finalized = {}
    for level in LEVELS:
        count = scores[level]["count"]
        if count == 0 and level != "all":
            continue
        finalized[level] = {
            "count": count,
            "exact_match": (scores[level]["exact_match"] / count) if count else 0.0,
        }
    return finalized


def _normalize_sql_for_match(sql: Dict, schema: Schema, kmap: Dict) -> Dict:
    valid_col_units = spider_eval.build_valid_col_units(sql["from"]["table_units"], schema)
    sql = spider_eval.rebuild_sql_val(sql)
    return spider_eval.rebuild_sql_col(valid_col_units, sql, kmap)


def run_exact_match_evaluation(
    gold_data: List[Tuple[str, str]],
    predictions: List[str],
    db_dir: str,
    table_path: str,
) -> Tuple[Dict, List[Dict]]:
    """Compute Spider exact-match scores and aligned per-example details."""
    if len(gold_data) != len(predictions):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != gold count ({len(gold_data)})"
        )

    kmaps = spider_eval.build_foreign_key_map_from_json(table_path)
    evaluator = spider_eval.Evaluator()
    scores = _init_exact_scores()
    details = []

    for idx, ((gold_sql, db_id), pred_sql) in enumerate(zip(gold_data, predictions)):
        db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
        schema = Schema(get_schema(str(db_path)))
        gold_parsed = get_sql(schema, gold_sql)
        hardness = evaluator.eval_hardness(gold_parsed)

        try:
            pred_parsed = get_sql(schema, pred_sql)
            parse_error = None
        except Exception as exc:
            pred_parsed = json.loads(json.dumps(EMPTY_SQL))
            parse_error = f"{type(exc).__name__}: {exc}"

        kmap = kmaps[db_id]
        gold_norm = _normalize_sql_for_match(gold_parsed, schema, kmap)
        pred_norm = _normalize_sql_for_match(pred_parsed, schema, kmap)
        exact = bool(evaluator.eval_exact_match(pred_norm, gold_norm))

        scores[hardness]["count"] += 1
        scores["all"]["count"] += 1
        scores[hardness]["exact_match"] += 1.0 if exact else 0.0
        scores["all"]["exact_match"] += 1.0 if exact else 0.0

        item = {
            "id": idx,
            "db_id": db_id,
            "hardness": hardness,
            "exact_match": exact,
        }
        if parse_error:
            item["error"] = parse_error
        details.append(item)

    return _finalize_exact_scores(scores), details


def execute_sql_with_timeout(db_path: str, sql: str, timeout_seconds: float) -> Dict:
    """Execute one SQLite query with a deadline and return rows or an error record."""
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    conn = None

    def progress_handler():
        nonlocal timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        return 0

    try:
        conn = sqlite3.connect(db_path, timeout=timeout_seconds)
        conn.set_progress_handler(progress_handler, 1000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return {"rows": rows, "error": None, "timeout": False}
    except Exception as exc:
        is_timeout = timed_out or (time.monotonic() >= deadline and "interrupted" in str(exc).lower())
        return {
            "rows": None,
            "error": f"{type(exc).__name__}: {exc}",
            "timeout": bool(is_timeout),
        }
    finally:
        if conn is not None:
            conn.set_progress_handler(None, 0)
            conn.close()


def _result_map(rows: List[Tuple], val_units: List) -> Dict:
    mapped = {}
    for idx, val_unit in enumerate(val_units):
        key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
        mapped[key] = [row[idx] for row in rows]
    return mapped


def _compare_exec_results(pred_rows: List[Tuple], gold_rows: List[Tuple], pred_sql: Dict, gold_sql: Dict) -> bool:
    pred_val_units = [unit[1] for unit in pred_sql["select"][1]]
    gold_val_units = [unit[1] for unit in gold_sql["select"][1]]
    return _result_map(pred_rows, pred_val_units) == _result_map(gold_rows, gold_val_units)


def run_execution_evaluation(
    gold_data: List[Tuple[str, str]],
    predictions: List[str],
    db_dir: str,
    table_path: str,
    timeout_seconds: float = 30.0,
) -> Tuple[Dict, List[Dict]]:
    """Compute execution accuracy with per-query timeout and per-example records."""
    if len(gold_data) != len(predictions):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != gold count ({len(gold_data)})"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    kmaps = spider_eval.build_foreign_key_map_from_json(table_path)
    details = []
    matches = 0

    for idx, ((gold_sql, db_id), pred_sql) in enumerate(zip(gold_data, predictions)):
        db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
        errors = []
        timed_out = False
        exec_match = False

        pred_result = execute_sql_with_timeout(str(db_path), pred_sql, timeout_seconds)
        gold_result = execute_sql_with_timeout(str(db_path), gold_sql, timeout_seconds)

        if pred_result["error"]:
            errors.append(f"pred_sql: {pred_result['error']}")
        if gold_result["error"]:
            errors.append(f"gold_sql: {gold_result['error']}")
        timed_out = bool(pred_result["timeout"] or gold_result["timeout"])

        if not errors:
            try:
                schema = Schema(get_schema(str(db_path)))
                kmap = kmaps[db_id]
                pred_parsed = _normalize_sql_for_match(get_sql(schema, pred_sql), schema, kmap)
                gold_parsed = _normalize_sql_for_match(get_sql(schema, gold_sql), schema, kmap)
                exec_match = _compare_exec_results(
                    pred_result["rows"],
                    gold_result["rows"],
                    pred_parsed,
                    gold_parsed,
                )
            except Exception as exc:
                errors.append(f"parse_or_compare: {type(exc).__name__}: {exc}")
                exec_match = False

        if exec_match:
            matches += 1

        details.append({
            "id": idx,
            "db_id": db_id,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "exec_match": exec_match,
            "error": "; ".join(errors) if errors else None,
            "timeout": timed_out,
        })

    count = len(details)
    scores = {
        "all": {
            "count": count,
            "execution_accuracy": (matches / count) if count else 0.0,
        }
    }
    return scores, details


def save_results(
    predictions: List[Dict],
    scores: Dict,
    meta: Dict,
    output_path: str
):
    """Save legacy combined evaluation results to JSON."""
    output_data = {
        "summary": {
            "count": scores["all"]["count"],
            "exact_match": scores["all"].get("exact_match"),
            "execution_accuracy": scores["all"].get("execution_accuracy"),
            "model": meta.get("model", "unknown"),
            "dataset": meta.get("dataset", "unknown"),
            "timestamp": meta.get("timestamp", datetime.now().isoformat())
        },
        "by_difficulty": {
            level: {
                "count": scores[level]["count"],
                "exact_match": scores[level].get("exact_match"),
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
    """Parse a gold file with lines formatted as `{sql}\t{db_id}`."""
    gold_data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Malformed gold line {line_num} in {gold_path}")
            gold_data.append((parts[0], parts[1]))
    return gold_data


def parse_pred_file(pred_path: str) -> List[str]:
    """Parse predictions while preserving blank prediction lines as entries."""
    with open(pred_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
