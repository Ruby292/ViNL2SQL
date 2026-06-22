"""Prompt builder for schema-filtered NL2SQL generation."""

from __future__ import annotations

from typing import Dict, List


def _format_table(table_name: str, columns: List[str], candidate_fk_edges: List[Dict]) -> str:
    lines = [f"Table: {table_name}"]
    fk_map = {
        edge["col"]: f"{edge['to']}.{edge['ref_col']}"
        for edge in candidate_fk_edges
        if edge["from"] == table_name
    }
    for column in columns:
        if column in fk_map:
            lines.append(f"  - {column} (FK -> {fk_map[column]})")
        else:
            lines.append(f"  - {column}")
    return "\n".join(lines)


def build_prompt(question: str, candidate_schema: Dict, db_id: str) -> str:
    tables = candidate_schema.get("candidate_tables", [])
    columns = candidate_schema.get("candidate_columns", {})
    fk_edges = candidate_schema.get("candidate_fk_edges", [])

    schema_lines = [
        "You are an expert SQL developer. Given a database schema and a natural language question, generate a valid SQL query.",
        "",
        f"Database: {db_id}",
        "",
    ]

    for table_name in tables:
        schema_lines.append(_format_table(table_name, columns.get(table_name, []), fk_edges))
        schema_lines.append("")

    schema_lines.extend([
        f"Question: {question}",
        "",
        "Generate only the SQL query without any explanation.",
        "The query should be syntactically correct and answer the question.",
    ])

    return "\n".join(schema_lines).strip()
