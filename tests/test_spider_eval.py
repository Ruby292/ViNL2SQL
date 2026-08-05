import tempfile
import unittest
import sqlite3
from pathlib import Path

from shared.spider_eval import (
    execute_sql_with_timeout,
    parse_gold_file,
    parse_pred_file,
    run_execution_evaluation,
)


class SpiderEvalTests(unittest.TestCase):
    def test_parse_pred_file_preserves_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.txt"
            path.write_text("SELECT 1\n\nSELECT 2\n", encoding="utf-8")
            self.assertEqual(parse_pred_file(str(path)), ["SELECT 1", "", "SELECT 2"])

    def test_parse_gold_file_uses_last_tab_as_db_separator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.txt"
            path.write_text("SELECT 'a\tb'\tdb1\n", encoding="utf-8")
            self.assertEqual(parse_gold_file(str(path)), [("SELECT 'a\tb'", "db1")])

    def test_parse_files_strip_utf8_bom_from_first_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold_path = Path(tmp) / "gold.txt"
            pred_path = Path(tmp) / "predictions.txt"
            gold_path.write_text("\ufeffSELECT 1\tdb1\n", encoding="utf-8")
            pred_path.write_text("\ufeffSELECT 1\n", encoding="utf-8")

            self.assertEqual(parse_gold_file(str(gold_path)), [("SELECT 1", "db1")])
            self.assertEqual(parse_pred_file(str(pred_path)), ["SELECT 1"])

    def test_execute_sql_reports_sql_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.sqlite"
            sqlite3.connect(db).close()
            result = execute_sql_with_timeout(str(db), "SELECT missing_column", 1)
            self.assertIsNone(result["rows"])
            self.assertFalse(result["timeout"])
            self.assertIn("OperationalError", result["error"])

    def test_run_execution_evaluation_records_match_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_dir = root / "db1"
            db_dir.mkdir()
            db_path = db_dir / "db1.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
            conn.commit()
            conn.close()

            table_path = root / "tables.json"
            table_path.write_text(
                """
                [{
                  "db_id": "db1",
                  "table_names_original": ["t"],
                  "column_names_original": [[-1, "*"], [0, "id"]],
                  "column_types": ["text", "number"],
                  "primary_keys": [],
                  "foreign_keys": []
                }]
                """,
                encoding="utf-8",
            )

            gold = [("SELECT COUNT(*) FROM t", "db1"), ("SELECT COUNT(*) FROM t", "db1")]
            preds = ["SELECT COUNT(*) FROM t", "SELECT missing_column FROM t"]
            scores, details = run_execution_evaluation(
                gold, preds, str(root), str(table_path), timeout_seconds=1
            )

            self.assertEqual(scores["all"]["count"], 2)
            self.assertEqual(scores["all"]["execution_accuracy"], 0.5)
            self.assertTrue(details[0]["exec_match"])
            self.assertIsNone(details[0]["error"])
            self.assertFalse(details[0]["timeout"])
            self.assertFalse(details[1]["exec_match"])
            self.assertIn("pred_sql", details[1]["error"])
            self.assertFalse(details[1]["timeout"])


if __name__ == "__main__":
    unittest.main()
