import re
from typing import Dict, List, Set


def clean_identifier(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return " ".join(name.lower().split())


def is_id_column(column_name: str) -> bool:
    """Return True for id/key-like columns that should not enrich table text."""
    normalized = column_name.strip().lower()
    return normalized == "id" or normalized.endswith("_id")


def is_internal_table(table_name: str) -> bool:
    """Return True for SQLite/system tables that should not appear in prompts."""
    return table_name.strip().lower().startswith("sqlite_")


def _looks_like_code_or_id(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or "code" in normalized
    )


def _looks_like_description(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return (
        "description" in normalized
        or normalized.endswith("_desc")
        or normalized.endswith("_name")
        or normalized == "name"
    )


def classify_table(table_name: str, columns: List[str], foreign_keys: List[Dict]) -> str:
    """Classify a table as entity, reference, or junction."""
    normalized_name = table_name.strip().lower()
    outgoing_fk_columns = {
        fk.get("column")
        for fk in foreign_keys
        if fk.get("column")
    }

    if (
        normalized_name.startswith("ref_")
        or normalized_name.endswith("_types")
        or normalized_name.endswith("_codes")
    ):
        return "reference"

    if len(columns) == 2 and not outgoing_fk_columns:
        first, second = columns
        if (
            (_looks_like_code_or_id(first) and _looks_like_description(second))
            or (_looks_like_code_or_id(second) and _looks_like_description(first))
        ):
            return "reference"

    if len(outgoing_fk_columns) >= 2:
        non_fk_columns = [
            column
            for column in columns
            if column not in outgoing_fk_columns
        ]
        if len(non_fk_columns) <= 1:
            return "junction"

    return "entity"


def get_table_priority(table_type: str) -> float:
    return {
        "entity": 1.0,
        "junction": 0.7,
        "reference": 0.6,
    }.get(table_type, 1.0)


def schema_name_set(db_schema: Dict) -> Set[str]:
    """Return lowercase table and column names for one Spider schema."""
    names = set()
    for table_name in db_schema["table_names_original"]:
        if not is_internal_table(table_name):
            names.add(table_name.lower())
    for table_idx, column_name in db_schema["column_names_original"]:
        if table_idx == -1:
            continue
        if not is_internal_table(db_schema["table_names_original"][table_idx]):
            names.add(column_name.lower())
    return names


def foreign_keys_by_table(db_schema: Dict) -> Dict[int, List[Dict]]:
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    by_table: Dict[int, List[Dict]] = {
        table_idx: []
        for table_idx in range(len(table_names))
    }
    for child_idx, parent_idx in db_schema.get("foreign_keys", []):
        child_table_idx, child_col = column_names[child_idx]
        parent_table_idx, parent_col = column_names[parent_idx]
        if child_table_idx < 0 or parent_table_idx < 0:
            continue
        by_table[child_table_idx].append(
            {
                "column": child_col,
                "ref_table": table_names[parent_table_idx],
                "ref_column": parent_col,
            }
        )
    return by_table


def extract_schema_nouns(db_schema: Dict) -> Dict[str, str]:
    """Build table-level schema texts for semantic matching.

    Each output key is the original table name. Each value is a richer table
    description made from the table name plus non-id column names, for example:
    ``singer: Name, Country, Song_Name, Age``.
    """
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]

    columns_by_table: Dict[int, List[str]] = {
        table_idx: []
        for table_idx in range(len(table_names))
    }

    for table_idx, column_name in column_names:
        if table_idx == -1:
            continue
        if is_id_column(column_name):
            continue
        columns_by_table[table_idx].append(column_name)

    table_texts = {}
    for table_idx, table_name in enumerate(table_names):
        if is_internal_table(table_name):
            continue
        columns = columns_by_table[table_idx]
        if columns:
            table_texts[table_name] = f"{table_name}: {', '.join(columns)}"
        else:
            table_texts[table_name] = table_name

    return table_texts
