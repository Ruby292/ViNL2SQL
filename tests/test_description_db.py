import json
import unittest
from pathlib import Path

from augmentation.schema_nouns import is_internal_table
from zero_shot.common import load_tables


BASE_DIR = Path(__file__).resolve().parents[1]
DESCRIPTION_DIR = BASE_DIR / "descriptions" / "db_descriptions"
DESCRIPTION_DB = DESCRIPTION_DIR / "description_db.json"
SCHEMA_DESCRIPTION_20DB = DESCRIPTION_DIR / "schema_description_20db.json"


class DescriptionDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        description_path = DESCRIPTION_DB if DESCRIPTION_DB.exists() else SCHEMA_DESCRIPTION_20DB
        cls.descriptions = json.loads(description_path.read_text(encoding="utf-8"))
        cls.tables = load_tables()

    def test_every_non_internal_schema_column_has_description(self):
        missing = []
        extra = []

        for db_id, description in self.descriptions.items():
            schema = self.tables[db_id]
            for table_idx, table_name in enumerate(schema["table_names_original"]):
                if is_internal_table(table_name):
                    continue

                table_info = description["tables"].get(table_name)
                self.assertIsNotNone(table_info, f"Missing table description: {db_id}.{table_name}")

                expected_columns = {
                    column_name
                    for column_table_idx, column_name in schema["column_names_original"]
                    if column_table_idx == table_idx
                }
                actual_columns = set(table_info.get("columns", {}))
                missing.extend(
                    (db_id, table_name, column_name)
                    for column_name in sorted(expected_columns - actual_columns)
                )
                extra.extend(
                    (db_id, table_name, column_name)
                    for column_name in sorted(actual_columns - expected_columns)
                )

        self.assertEqual([], missing)
        self.assertEqual([], extra)

    def test_table_entries_are_complete_for_prompting(self):
        empty_descriptions = []
        empty_columns = []
        invalid_join_hints = []

        for db_id, description in self.descriptions.items():
            if not isinstance(description.get("join_hints", []), list):
                invalid_join_hints.append(db_id)
            for table_name, table_info in description.get("tables", {}).items():
                if not table_info.get("description"):
                    empty_descriptions.append((db_id, table_name))
                if not table_info.get("columns"):
                    empty_columns.append((db_id, table_name))

        self.assertEqual([], empty_descriptions)
        self.assertEqual([], empty_columns)
        self.assertEqual([], invalid_join_hints)


if __name__ == "__main__":
    unittest.main()
