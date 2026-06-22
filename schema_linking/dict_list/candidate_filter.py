"""Schema filtering based on a precomputed dictionary list."""

from __future__ import annotations

import re
from typing import Dict, List


MIN_TABLES = 3
MAX_TABLES = 15
TOP_TABLES = 5
TOP_COLUMNS = 8
GENERIC_TERMS = {"name", "id", "type", "code", "date", "time", "value", "num"}


def normalize_question(question: str) -> List[str]:
    text = question.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [tok for tok in text.split() if tok]
    ngrams: List[str] = []
    for n in range(1, 4):
        for idx in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[idx : idx + n]))
    return ngrams or tokens


def _score_terms(ngrams: List[str], terms: Dict[str, List[str]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for phrase in ngrams:
        if phrase not in terms:
            continue
        weight = 1.0 if phrase not in GENERIC_TERMS else 0.25
        for target in terms[phrase]:
            scores[target] = scores.get(target, 0.0) + weight
    return scores


def _table_index(db_entry: Dict) -> Dict[str, Dict]:
    return {table["name"]: table for table in db_entry["tables"]}


def _table_score(table: Dict, terms_scores: Dict[str, float], question_ngrams: List[str]) -> float:
    score = 0.0
    table_name = table["name"]

    if table_name.lower() in question_ngrams:
        score += 3.0
    for alias in table.get("aliases", []):
        if alias in question_ngrams:
            score += 2.0

    score += terms_scores.get(table_name, 0.0)

    for column in table.get("columns", []):
        score += 0.5 * terms_scores.get(f"{table_name}.{column}", 0.0)

    return score


def _column_score(table_name: str, column: str, terms_scores: Dict[str, float], question_ngrams: List[str]) -> float:
    key = f"{table_name}.{column}"
    score = terms_scores.get(key, 0.0)
    if column.lower() in question_ngrams:
        score += 1.0
    if column.lower() in GENERIC_TERMS:
        score *= 0.5
    return score


def filter_schema(question: str, db_id: str, dict_list: Dict[str, Dict]) -> Dict:
    if db_id not in dict_list:
        raise ValueError(f"Unknown db_id: {db_id}")

    db_entry = dict_list[db_id]
    tables = db_entry["tables"]
    table_map = _table_index(db_entry)
    ngrams = normalize_question(question)
    terms_scores = _score_terms(ngrams, db_entry.get("terms", {}))

    table_scores: Dict[str, float] = {}
    column_scores: Dict[str, Dict[str, float]] = {}

    for table in tables:
        table_name = table["name"]
        score = _table_score(table, terms_scores, ngrams)
        for column in table.get("columns", []):
            column_scores.setdefault(table_name, {})[column] = _column_score(table_name, column, terms_scores, ngrams)
        if score > 0:
            table_scores[table_name] = score

    if not table_scores:
        selected_tables = tables[:]
    else:
        selected_names = [name for name, _ in sorted(table_scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_TABLES]]
        selected_tables = [table_map[name] for name in selected_names if name in table_map]

        neighbors = []
        for table_name in selected_names:
            for neighbor in db_entry.get("fk_graph", {}).get(table_name, []):
                if neighbor not in selected_names and neighbor in table_map:
                    neighbors.append(table_map[neighbor])
        selected_tables.extend(neighbors)

    selected_by_name = {}
    for table in selected_tables:
        selected_by_name[table["name"]] = table

    if len(selected_by_name) < 2:
        selected_by_name = table_map

    candidate_tables = []
    candidate_columns: Dict[str, List[str]] = {}
    candidate_fk_edges = []
    scores: Dict[str, float] = {}

    for table_name, table in selected_by_name.items():
        table_score = table_scores.get(table_name, 0.0)
        scores[table_name] = table_score
        candidate_tables.append(table_name)

        cols = []
        per_column_scores = column_scores.get(table_name, {})
        ranked_columns = sorted(
            table.get("columns", []),
            key=lambda col: (-per_column_scores.get(col, 0.0), col),
        )

        for col in ranked_columns:
            if per_column_scores.get(col, 0.0) > 0:
                cols.append(col)

        for col in table.get("primary_key", []):
            if col not in cols:
                cols.append(col)
        for fk in table.get("foreign_keys", []):
            col = fk["column"]
            if col not in cols:
                cols.append(col)
            candidate_fk_edges.append(
                {
                    "from": table_name,
                    "col": fk["column"],
                    "to": fk["ref_table"],
                    "ref_col": fk["ref_column"],
                }
            )

        if not cols:
            cols = table.get("columns", [])[:TOP_COLUMNS]

        candidate_columns[table_name] = cols[:TOP_COLUMNS]

    candidate_tables = sorted(dict.fromkeys(candidate_tables), key=lambda name: (-scores.get(name, 0.0), name))
    if len(candidate_tables) > MAX_TABLES:
        candidate_tables = candidate_tables[:MAX_TABLES]
        candidate_columns = {name: candidate_columns[name] for name in candidate_tables if name in candidate_columns}
        candidate_fk_edges = [edge for edge in candidate_fk_edges if edge["from"] in candidate_tables and edge["to"] in candidate_tables]
        scores = {name: scores[name] for name in candidate_tables if name in scores}

    return {
        "db_id": db_id,
        "candidate_tables": candidate_tables,
        "candidate_columns": candidate_columns,
        "candidate_fk_edges": candidate_fk_edges,
        "scores": scores,
        "fallback_full_schema": len(table_scores) == 0,
    }
