import json
from pathlib import Path
from typing import Dict

from augmentation.schema_nouns import is_internal_table


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTIONS_DIR = BASE_DIR / "descriptions" / "db_descriptions"
DESCRIPTION_POLICY_LINES = [
    "Quy ước mô tả:",
    "- Mọi cột trong schema đều có mô tả; cột rõ nghĩa được mô tả ngắn.",
    "- Cột khóa ngoại, cờ/mã, tên viết tắt, typo, hoặc kiểu dữ liệu dễ nhầm được giải thích rõ hơn.",
    "",
]


class DescriptionLoader:
    """Load and cache precomputed database descriptions.

    This class only reads JSON files. It does not call an LLM.
    """

    def __init__(self, descriptions_dir=DEFAULT_DESCRIPTIONS_DIR, description_file=None):
        path = Path(descriptions_dir)
        self.descriptions_dir = path if path.is_absolute() else BASE_DIR / path
        if description_file:
            file_path = Path(description_file)
            if file_path.is_absolute():
                self.description_file = file_path
            else:
                repo_relative = BASE_DIR / file_path
                self.description_file = (
                    repo_relative
                    if repo_relative.exists()
                    else self.descriptions_dir / file_path
                )
        else:
            self.description_file = None
        self._cache: Dict[str, Dict] = {}
        self._aggregate_cache = None

    @property
    def source_path(self) -> Path:
        return self.description_file or self.descriptions_dir / "description_db.json"

    def _load_aggregate(self) -> Dict:
        if self._aggregate_cache is not None:
            return self._aggregate_cache

        path = self.source_path
        if not path.exists():
            self._aggregate_cache = {}
            return self._aggregate_cache

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._aggregate_cache = data
        return self._aggregate_cache

    @staticmethod
    def _case_insensitive_get(data: Dict, key: str) -> Dict:
        if key in data:
            return data[key]

        normalized_key = key.lower()
        for candidate_key, value in data.items():
            if candidate_key.lower() == normalized_key:
                return value
        return {}

    def load(self, db_id: str) -> Dict:
        if db_id in self._cache:
            return self._cache[db_id]

        path = self.descriptions_dir / f"{db_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = self._case_insensitive_get(self._load_aggregate(), db_id)
        self._cache[db_id] = data
        return data

    def format_for_prompt(self, db_id: str) -> str:
        data = self.load(db_id)
        if not data:
            return ""

        lines = []
        db_description = data.get("db_description")
        if db_description:
            lines.extend(["Database description:", f"Mô tả database: {db_description}", ""])
            lines.extend(DESCRIPTION_POLICY_LINES)

        tables = data.get("tables", {})
        for table_name, table_info in tables.items():
            if is_internal_table(table_name):
                continue

            table_type = table_info.get("type")
            suffix = f" [{table_type}]" if table_type else ""
            lines.append(f"Bảng: {table_name}{suffix}")

            description = table_info.get("description")
            if description:
                lines.append(f"  Mô tả: {description}")

            columns = table_info.get("columns", {})
            if columns:
                lines.append("  Mô tả cột:")
                for column_name, column_description in columns.items():
                    lines.append(f"    - {column_name}: {column_description}")

            sample_values = table_info.get("sample_values", {})
            if sample_values:
                lines.append("  Giá trị mẫu:")
                for column_name, values in sample_values.items():
                    formatted_values = json.dumps(values, ensure_ascii=False)
                    lines.append(f"    - {column_name}: {formatted_values}")
            lines.append("")

        relationships = data.get("relationships", [])
        if relationships:
            lines.append("Quan hệ:")
            for relationship in relationships:
                lines.append(f"  - {relationship}")

        join_hints = data.get("join_hints", [])
        if join_hints:
            lines.append("Gợi ý join:")
            for join_hint in join_hints:
                lines.append(f"  - {join_hint}")

        return "\n".join(lines).strip()
