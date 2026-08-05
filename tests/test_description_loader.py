import json
import tempfile
import unittest
from pathlib import Path

from descriptions.loader import DescriptionLoader


class DescriptionLoaderTests(unittest.TestCase):
    def test_missing_description_returns_empty_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DescriptionLoader(tmpdir)

            self.assertEqual(loader.load("missing_db"), {})
            self.assertEqual(loader.format_for_prompt("missing_db"), "")

    def test_formats_description_and_skips_sqlite_internal_tables(self):
        payload = {
            "db_description": "Stores customers and purchases.",
            "tables": {
                "customer": {
                    "description": "Customer master data.",
                    "type": "main",
                    "columns": {"cust_code": "Customer business code."},
                    "sample_values": {"status": ["active", "inactive"]},
                },
                "sqlite_sequence": {
                    "description": "Internal SQLite table.",
                    "type": "internal",
                },
            },
            "relationships": [
                "purchase.customer_id -> customer.id (purchase owner)"
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shop.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            formatted = DescriptionLoader(tmpdir).format_for_prompt("shop")

        self.assertIn("Database description:", formatted)
        self.assertIn("Quy ước mô tả:", formatted)
        self.assertIn("Mọi cột trong schema đều có mô tả", formatted)
        self.assertIn("Stores customers and purchases.", formatted)
        self.assertIn("customer [main]", formatted)
        self.assertIn("Mô tả cột:", formatted)
        self.assertIn("cust_code: Customer business code.", formatted)
        self.assertIn('"active"', formatted)
        self.assertIn("purchase.customer_id -> customer.id", formatted)
        self.assertNotIn("sqlite_sequence", formatted)

    def test_loads_aggregate_description_file_case_insensitively(self):
        payload = {
            "WORLD_1": {
                "db_description": "World geography data.",
                "tables": {
                    "country": {
                        "description": "Countries and territories.",
                        "type": "main",
                    }
                },
                "relationships": [],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "description_db.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            formatted = DescriptionLoader(tmpdir).format_for_prompt("world_1")

        self.assertIn("World geography data.", formatted)
        self.assertIn("country [main]", formatted)

    def test_loads_explicit_aggregate_description_file_with_join_hints(self):
        payload = {
            "concert_singer": {
                "db_description": "Concert and singer data.",
                "tables": {
                    "concert": {
                        "description": "Concert events.",
                        "columns": {"concert_ID": "Primary key."},
                    }
                },
                "join_hints": [
                    "concert -> stadium: concert.Stadium_ID = stadium.Stadium_ID"
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema_description_20db.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            formatted = DescriptionLoader(
                descriptions_dir=tmpdir,
                description_file=path,
            ).format_for_prompt("concert_singer")

        self.assertIn("Concert and singer data.", formatted)
        self.assertIn("concert_ID: Primary key.", formatted)
        self.assertIn("Gợi ý join:", formatted)
        self.assertIn("concert.Stadium_ID = stadium.Stadium_ID", formatted)


if __name__ == "__main__":
    unittest.main()
