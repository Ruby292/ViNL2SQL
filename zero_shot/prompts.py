"""
Prompt formatting utilities for Text-to-SQL zero-shot inference.
"""

import re
from typing import Dict, List, Tuple

from augmentation.schema_nouns import is_internal_table


def format_schema(db_id: str, tables: Dict) -> str:
    """
    Format database schema from tables.json into readable text format.

    Args:
        db_id: Database identifier (e.g., "concert_singer")
        tables: Dict loaded from tables.json, keyed by db_id

    Returns:
        Formatted schema string with tables, columns, types, and foreign keys
    """
    if db_id not in tables:
        raise ValueError(f"Database '{db_id}' not found in tables.json")

    db_schema = tables[db_id]

    table_names_original = db_schema['table_names_original']
    column_names_original = db_schema['column_names_original']
    column_types = db_schema['column_types']
    foreign_keys = db_schema['foreign_keys']

    schema_lines = [f"Database: {db_id}", ""]

    # Group columns by table
    for table_idx, table_name in enumerate(table_names_original):
        if is_internal_table(table_name):
            continue
        schema_lines.append(f"Table: {table_name}")

        # Find all columns for this table
        for col_idx, (tbl_idx, col_name) in enumerate(column_names_original):
            # Skip placeholder column at index -1
            if tbl_idx == -1:
                continue

            if tbl_idx == table_idx:
                col_type = column_types[col_idx]
                schema_lines.append(f"  - {col_name} ({col_type})")

        schema_lines.append("")

    # Format foreign keys
    if foreign_keys:
        fk_lines = []
        for fk_pair in foreign_keys:
            from_col_idx, to_col_idx = fk_pair

            from_table_idx, from_col_name = column_names_original[from_col_idx]
            to_table_idx, to_col_name = column_names_original[to_col_idx]

            from_table_name = table_names_original[from_table_idx]
            to_table_name = table_names_original[to_table_idx]

            if is_internal_table(from_table_name) or is_internal_table(to_table_name):
                continue
            fk_lines.append(f"  - {from_table_name}.{from_col_name} -> {to_table_name}.{to_col_name}")
        if fk_lines:
            schema_lines.append("Foreign Keys:")
            schema_lines.extend(fk_lines)

    return "\n".join(schema_lines)


def _format_description_section(description_text: str = "") -> str:
    text = (description_text or "").strip()
    return f"{text}\n\n" if text else ""


def build_prompt(
    question: str,
    db_id: str,
    tables: Dict,
    description_text: str = "",
) -> str:
    """
    Build user prompt for SQL generation, including schema and instruction.

    Note: This returns the raw user prompt. Chat template formatting is applied
    in run_zero_shot.py using tokenizer.apply_chat_template().

    Args:
        question: Natural language question in Vietnamese
        db_id: Database identifier
        tables: Dict loaded from tables.json
        description_text: Optional precomputed database description

    Returns:
        User prompt string (without chat template formatting)
    """
    schema = format_schema(db_id, tables)
    description_section = _format_description_section(description_text)

    prompt = f"""You are an expert SQL developer. Given a database schema and a natural language question, generate a valid SQL query.

{description_section}{schema}

Question: {question}

Generate only the SQL query without any explanation.
The query should be syntactically correct and answer the question."""

    return prompt


def build_prompt_augmented(
    question: str,
    db_id: str,
    tables: Dict,
    hints: List[Dict],
    description_text: str = "",
) -> str:
    """Build the Text-to-SQL prompt with optional schema item hints."""
    if not hints:
        return build_prompt(question, db_id, tables, description_text=description_text)

    schema = format_schema(db_id, tables)
    description_section = _format_description_section(description_text)
    hint_lines = []
    for hint in hints:
        if hint.get("column"):
            hint_lines.append(f'  - "{hint["vi_noun"]}" → {hint["table"]}.{hint["column"]}')
        else:
            hint_lines.append(f'  - "{hint["vi_noun"]}" → table: {hint["table"]}')
    hints_section = (
        "\nThe following Vietnamese terms map to these schema items:\n"
        + "\n".join(hint_lines)
        + "\n"
    )

    prompt = f"""You are an expert SQL developer. Given a database schema and a natural language question, generate a valid SQL query.

{description_section}{schema}
{hints_section}
Question: {question}

Generate only the SQL query without any explanation.
The query should be syntactically correct and answer the question."""

    return prompt


def extract_sql(text: str) -> str:
    """
    Extract SQL query from model output, removing markdown fences and explanations.

    Args:
        text: Raw model output text

    Returns:
        Cleaned SQL query string
    """
    # Remove markdown code fences
    text = re.sub(r'```sql\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Try to find SQL statement (SELECT or WITH for CTEs)
    sql_pattern = r'((?:WITH|SELECT)\s+.+?)(?:;|\Z)'
    match = re.search(r'(?i)((?:WITH|SELECT)[\s\S]*?)(?:;|$)', text)

    if match:
        sql = match.group(1).strip()
        # Remove trailing semicolon if present
        sql = sql.rstrip(';').strip()
        return sql

    # Fallback: return cleaned text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text
