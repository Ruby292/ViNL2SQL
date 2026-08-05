import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from augmentation import similarity
from augmentation.similarity import (
    build_matching_targets,
    build_rich_targets,
    compute_matches,
    compute_target_matches,
    document_prefix_for_model,
    encode_texts,
    get_encoder,
    query_prefix_for_model,
)


MATCH_SCHEMA = {
    "table_names_original": [
        "singer",
        "concert",
        "singer_in_concert",
        "Ref_Template_Types",
    ],
    "column_names_original": [
        [-1, "*"],
        [0, "Singer_ID"],
        [0, "Name"],
        [0, "Country"],
        [1, "Concert_ID"],
        [1, "Venue"],
        [2, "Singer_ID"],
        [2, "Concert_ID"],
        [3, "Template_Type_ID"],
        [3, "Template_Type_Description"],
    ],
    "column_types": [
        "text",
        "number",
        "text",
        "text",
        "number",
        "text",
        "number",
        "number",
        "number",
        "text",
    ],
    "primary_keys": [1, 4, 8],
    "foreign_keys": [[6, 1], [7, 4]],
}


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        similarity._MODEL_CACHE.clear()

    def tearDown(self):
        similarity._MODEL_CACHE.clear()

    def test_compute_matches_returns_empty_for_empty_inputs(self):
        self.assertEqual(
            compute_matches([], np.empty((0, 2)), ["Singer"], np.ones((1, 2))),
            [],
        )
        self.assertEqual(
            compute_matches(["tên"], np.ones((1, 2)), [], np.empty((0, 2))),
            [],
        )

    def test_compute_matches_keeps_top_table_per_noun_sorted(self):
        noun_embs = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        table_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        matches = compute_matches(
            ["tên", "quốc gia"],
            noun_embs,
            ["Singer", "Country"],
            table_embs,
            threshold=0.75,
        )

        self.assertEqual(
            matches,
            [
                {"vi_noun": "tên", "table": "Singer", "similarity": 1.0},
                {"vi_noun": "quốc gia", "table": "Singer", "similarity": 0.8},
            ],
        )

    def test_compute_matches_discards_below_threshold(self):
        matches = compute_matches(
            ["tên"],
            np.array([[0.7, 0.0]], dtype=np.float32),
            ["Singer"],
            np.array([[1.0, 0.0]], dtype=np.float32),
            threshold=0.8,
        )
        self.assertEqual(matches, [])

    def test_build_matching_targets_adds_column_targets_and_skips_pure_ids(self):
        targets = build_matching_targets(MATCH_SCHEMA, "concert_singer")
        target_by_key = {
            (target["table"], target.get("column")): target
            for target in targets
        }

        self.assertEqual(target_by_key[("singer", None)]["text"], "singer")
        self.assertEqual(target_by_key[("singer", "Name")]["text"], "singer Name")
        self.assertEqual(target_by_key[("singer", "Name")]["table_type"], "entity")
        self.assertEqual(
            target_by_key[("singer_in_concert", None)]["table_type"],
            "junction",
        )
        self.assertEqual(
            target_by_key[("Ref_Template_Types", None)]["table_type"],
            "reference",
        )
        self.assertNotIn(("singer", "Singer_ID"), target_by_key)
        self.assertNotIn(("singer_in_concert", "Singer_ID"), target_by_key)
        self.assertNotIn(("Ref_Template_Types", "Template_Type_ID"), target_by_key)

    def test_build_rich_targets_uses_descriptions_and_skips_pure_pk_descriptions(self):
        schema_desc = {
            "concert_singer": {
                "tables": {
                    "singer": {
                        "description": "Thông tin ca sĩ.",
                        "columns": {
                            "Singer_ID": "PK. Định danh ca sĩ.",
                            "Name": "Tên ca sĩ.",
                            "Country": "Quốc gia của ca sĩ.",
                        },
                    },
                    "concert": {
                        "description": "Thông tin buổi hòa nhạc.",
                        "columns": {
                            "Concert_ID": "PK. Định danh concert.",
                            "Venue": "Địa điểm tổ chức.",
                        },
                    },
                },
            }
        }

        targets = build_rich_targets(schema_desc, "concert_singer", MATCH_SCHEMA)
        target_by_key = {
            (target["table"], target.get("column")): target
            for target in targets
        }

        self.assertIn("Thông tin ca sĩ.", target_by_key[("singer", None)]["text"])
        self.assertEqual(
            target_by_key[("singer", "Name")]["text"],
            "singer Name Tên ca sĩ.",
        )
        self.assertNotIn(("singer", "Singer_ID"), target_by_key)
        self.assertNotIn(("concert", "Concert_ID"), target_by_key)

    def test_build_rich_targets_falls_back_to_raw_targets_for_missing_db(self):
        targets = build_rich_targets({}, "concert_singer", MATCH_SCHEMA)
        target_by_key = {
            (target["table"], target.get("column")): target
            for target in targets
        }

        self.assertEqual(target_by_key[("singer", None)]["text"], "singer")
        self.assertEqual(target_by_key[("singer", "Name")]["text"], "singer Name")

    def test_compute_target_matches_applies_table_type_priority(self):
        targets = [
            {
                "text": "singer Name",
                "table": "singer",
                "column": "Name",
                "table_type": "entity",
            },
            {
                "text": "Ref_Template_Types Template_Type_Description",
                "table": "Ref_Template_Types",
                "column": "Template_Type_Description",
                "table_type": "reference",
            },
        ]
        noun_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        target_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        matches = compute_target_matches(
            ["tên", "loại"],
            noun_embs,
            targets,
            target_embs,
            threshold=0.75,
        )

        self.assertEqual(
            matches,
            [
                {
                    "vi_noun": "tên",
                    "table": "singer",
                    "column": "Name",
                    "similarity": 1.0,
                    "table_type": "entity",
                }
            ],
        )

    def test_prefixes_follow_e5_query_and_passage_format(self):
        e5_model = "intfloat/multilingual-e5-large-instruct"

        self.assertEqual(query_prefix_for_model(e5_model), "query: ")
        self.assertEqual(document_prefix_for_model(e5_model), "passage: ")
        self.assertEqual(query_prefix_for_model("google/embeddinggemma-300m"), "")
        self.assertEqual(document_prefix_for_model("google/embeddinggemma-300m"), "")

    def test_get_encoder_caches_sentence_transformer(self):
        created = []

        class FakeSentenceTransformer:
            def __init__(self, model_name):
                created.append(model_name)

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            first = get_encoder("fake-model")
            second = get_encoder("fake-model")

        self.assertIs(first, second)
        self.assertEqual(created, ["fake-model"])

    def test_encode_texts_uses_embeddinggemma_query_and_document_methods(self):
        calls = []

        class FakeEmbeddingGemma:
            def encode_query(self, texts, **kwargs):
                calls.append(("query", texts, kwargs))
                return np.array([[1.0, 0.0]], dtype=np.float32)

            def encode_document(self, texts, **kwargs):
                calls.append(("document", texts, kwargs))
                return np.array([[0.0, 1.0]], dtype=np.float32)

        encoder = FakeEmbeddingGemma()

        query_embs = encode_texts(
            ["stadium"],
            encoder,
            task="query",
            model_name="google/embeddinggemma-300m",
        )
        document_embs = encode_texts(
            ["stadium: Name, Location"],
            encoder,
            task="document",
            model_name="google/embeddinggemma-300m",
        )

        self.assertEqual(calls[0][0], "query")
        self.assertEqual(calls[0][1], ["stadium"])
        self.assertTrue(calls[0][2]["normalize_embeddings"])
        self.assertEqual(calls[1][0], "document")
        self.assertEqual(calls[1][1], ["stadium: Name, Location"])
        self.assertTrue(calls[1][2]["normalize_embeddings"])
        self.assertEqual(query_embs.dtype, np.float32)
        self.assertEqual(document_embs.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
