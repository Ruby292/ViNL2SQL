import re
from typing import Dict


def clean_identifier(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return " ".join(name.lower().split())


def extract_schema_nouns(db_schema: Dict) -> Dict[str, str]:
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]

    schema_nouns = {}
    for table_name in table_names:
        schema_nouns[table_name] = clean_identifier(table_name)

    for table_idx, column_name in column_names:
        if table_idx == -1:
            continue
        table_name = table_names[table_idx]
        display_key = f"{table_name}.{column_name}"
        schema_nouns[display_key] = (
            f"{clean_identifier(table_name)} {clean_identifier(column_name)}"
        )

    return schema_nouns
