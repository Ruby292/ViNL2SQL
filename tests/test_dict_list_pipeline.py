import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from schema_linking.dict_list.candidate_filter import filter_schema
from schema_linking.dict_list.prompt_builder import build_prompt
from schema_linking.dict_list.run_pipeline import load_dict_list


BASE_DIR = Path(__file__).resolve().parents[1]
DICT_LIST_PATH = BASE_DIR / "schema_linking" / "dict_list" / "results" / "dict_list.json"
DEV_PATH = BASE_DIR / "data" / "vispider_data" / "vispider_dev.json"


class DictListPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DICT_LIST_PATH.exists():
            raise unittest.SkipTest("dict_list.json not found; run to_dict_list.py first")
        cls.dict_list = load_dict_list(str(DICT_LIST_PATH))
        with open(DEV_PATH, encoding="utf-8") as f:
            cls.examples = json.load(f)

    def test_filter_schema_returns_candidate_schema(self):
        example = self.examples[0]
        candidate = filter_schema(example["question_vi"], example["db_id"], self.dict_list)

        self.assertEqual(candidate["db_id"], example["db_id"])
        self.assertIn("candidate_tables", candidate)
        self.assertIn("candidate_columns", candidate)
        self.assertIn("candidate_fk_edges", candidate)
        self.assertIn("scores", candidate)
        self.assertGreater(len(candidate["candidate_tables"]), 0)

    def test_prompt_builder_includes_question_and_schema(self):
        example = self.examples[0]
        candidate = filter_schema(example["question_vi"], example["db_id"], self.dict_list)
        prompt = build_prompt(example["question_vi"], candidate, example["db_id"])

        self.assertIn(example["question_vi"], prompt)
        self.assertIn(f"Database: {example['db_id']}", prompt)
        self.assertIn("Generate only the SQL query", prompt)

    def test_filter_schema_falls_back_safely(self):
        example = self.examples[0]
        candidate = filter_schema("completely unrelated query text", example["db_id"], self.dict_list)

        self.assertTrue(candidate["fallback_full_schema"])
        self.assertGreater(len(candidate["candidate_tables"]), 0)


class PipelineMockTest(unittest.TestCase):
    @patch("schema_linking.dict_list.run_pipeline.load_json")
    @patch("schema_linking.dict_list.run_pipeline.load_dict_list")
    @patch("schema_linking.dict_list.run_pipeline.load_model")
    @patch("schema_linking.dict_list.run_pipeline.batch_call_qwen")
    @patch("schema_linking.dict_list.run_pipeline.run_evaluation")
    @patch("schema_linking.dict_list.run_pipeline.save_results")
    def test_mock_pipeline_flow(
        self,
        save_results_mock,
        run_evaluation_mock,
        batch_call_qwen_mock,
        load_model_mock,
        load_dict_list_mock,
        load_json_mock,
    ):
        with open(DEV_PATH, encoding="utf-8") as f:
            example = json.load(f)[0]
        load_json_mock.side_effect = [[example], [example]]
        load_dict_list_mock.return_value = {
            example["db_id"]: {
                "db_id": example["db_id"],
                "tables": [],
                "terms": {},
                "fk_graph": {},
            }
        }
        load_model_mock.return_value = (object(), object())
        batch_call_qwen_mock.return_value = [example["query"]]
        run_evaluation_mock.return_value = {
            "all": {"count": 1, "exact_match": 1.0, "execution_accuracy": 1.0},
            "easy": {"count": 1, "exact_match": 1.0, "execution_accuracy": 1.0},
            "medium": {"count": 0, "exact_match": 0.0, "execution_accuracy": 0.0},
            "hard": {"count": 0, "exact_match": 0.0, "execution_accuracy": 0.0},
            "extra": {"count": 0, "exact_match": 0.0, "execution_accuracy": 0.0},
        }

        from schema_linking.dict_list import run_pipeline

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            args = [
                "--dev", str(DEV_PATH),
                "--dict_list", str(DICT_LIST_PATH),
                "--output_dir", str(output_dir),
                "--backend", "vllm",
                "--limit", "1",
            ]
            with patch("sys.argv", ["run_pipeline.py", *args]):
                run_pipeline.main()

        self.assertTrue(save_results_mock.called)
        self.assertTrue(run_evaluation_mock.called)
        self.assertTrue(batch_call_qwen_mock.called)


if __name__ == "__main__":
    unittest.main()
