import re
from typing import Dict, List


def clean_identifier(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return " ".join(name.lower().split())


def is_id_column(column_name: str) -> bool:
    """Return True for id/key-like columns that should not enrich table text."""
    normalized = column_name.strip().lower()
    return normalized == "id" or normalized.endswith("_id")


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
        columns = columns_by_table[table_idx]
        if columns:
            table_texts[table_name] = f"{table_name}: {', '.join(columns)}"
        else:
            table_texts[table_name] = table_name

    return table_texts
