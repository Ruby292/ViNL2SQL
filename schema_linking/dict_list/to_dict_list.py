"""Build a compact schema dictionary from Spider databases."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List


SYNONYM_MAP = {
    "publication": ["paper", "article", "work"],
    "author": ["writer", "researcher"],
    "organization": ["org", "institution", "company"],
    "conference": ["conf", "event", "venue"],
    "journal": ["journal", "periodical"],
    "student": ["learner", "pupil"],
    "employee": ["staff", "worker"],
    "department": ["dept", "division"],
}


def split_terms(name: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    tokens: List[str] = []
    for part in parts:
        if not part:
            continue
        camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part).split()
        for item in camel:
            lower = item.lower()
            if lower:
                tokens.append(lower)
    return tokens or [name.lower()]


def aliases_for(name: str) -> List[str]:
    base = [name.lower()]
    base.extend(split_terms(name))
    base.extend(SYNONYM_MAP.get(name.lower(), []))
    seen = set()
    aliases = []
    for item in base:
        item = item.strip().lower()
        if item and item not in seen:
            seen.add(item)
            aliases.append(item)
    return aliases


def make_terms(tables: List[Dict]) -> Dict[str, List[str]]:
    terms: Dict[str, List[str]] = {}
    for table in tables:
        table_name = table["name"]
        table_targets = [table_name]
        for alias in table["aliases"]:
            terms.setdefault(alias, [])
            for target in table_targets:
                if target not in terms[alias]:
                    terms[alias].append(target)

        for column in table["columns"]:
            column_key = f"{table_name}.{column}"
            for phrase in {column.lower(), *split_terms(column)}:
                terms.setdefault(phrase, [])
                if column_key not in terms[phrase]:
                    terms[phrase].append(column_key)
                if table_name not in terms[phrase]:
                    terms[phrase].append(table_name)
    return terms


def fetch_schema(sqlite_path: Path) -> Dict:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        table_names = [row[0] for row in cursor.fetchall()]

        tables: List[Dict] = []
        fk_graph: Dict[str, List[str]] = {}

        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info({table_name})")
            rows = cursor.fetchall()

            columns = [str(row[1]) for row in rows]
            primary_key = [str(row[1]) for row in rows if row[5] != 0]

            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fk_rows = cursor.fetchall()
            foreign_keys = []
            for fk in fk_rows:
                ref_table = str(fk[2])
                fk_entry = {
                    "column": str(fk[3]),
                    "ref_table": ref_table,
                    "ref_column": str(fk[4]),
                }
                foreign_keys.append(fk_entry)
                fk_graph.setdefault(table_name, [])
                if ref_table not in fk_graph[table_name]:
                    fk_graph[table_name].append(ref_table)
                fk_graph.setdefault(ref_table, [])
                if table_name not in fk_graph[ref_table]:
                    fk_graph[ref_table].append(table_name)

            table = {
                "name": table_name,
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "aliases": aliases_for(table_name),
            }
            tables.append(table)

        terms = make_terms(tables)
        return {
            "db_id": sqlite_path.stem,
            "db_path": str(sqlite_path).replace("\\", "/"),
            "tables": tables,
            "terms": terms,
            "fk_graph": fk_graph,
        }
    finally:
        conn.close()


def build_dict_list(db_root: Path) -> List[Dict]:
    records = []
    for db_dir in sorted(p for p in db_root.iterdir() if p.is_dir()):
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        if sqlite_path.exists():
            records.append(fetch_schema(sqlite_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Spider schema dictionary list")
    parser.add_argument("--db_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_root = Path(args.db_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = build_dict_list(db_root)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} database schemas to {output}")


if __name__ == "__main__":
    main()
