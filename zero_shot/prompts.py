"""
Prompt formatting utilities for Text-to-SQL zero-shot inference.
"""

import re
from typing import Dict, List, Tuple


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
        schema_lines.append("Foreign Keys:")
        for fk_pair in foreign_keys:
            from_col_idx, to_col_idx = fk_pair

            from_table_idx, from_col_name = column_names_original[from_col_idx]
            to_table_idx, to_col_name = column_names_original[to_col_idx]

            from_table_name = table_names_original[from_table_idx]
            to_table_name = table_names_original[to_table_idx]

            schema_lines.append(f"  - {from_table_name}.{from_col_name} -> {to_table_name}.{to_col_name}")

    return "\n".join(schema_lines)


def build_prompt(question: str, db_id: str, tables: Dict) -> str:
    """
    Build user prompt for SQL generation, including schema and instruction.

    Note: This returns the raw user prompt. Chat template formatting is applied
    in run_zero_shot.py using tokenizer.apply_chat_template().

    Args:
        question: Natural language question in Vietnamese
        db_id: Database identifier
        tables: Dict loaded from tables.json

    Returns:
        User prompt string (without chat template formatting)
    """
    schema = format_schema(db_id, tables)

    prompt = f"""You are an expert SQL developer. Given a database schema and a natural language question, generate a valid SQL query.

{schema}

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
