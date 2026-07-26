import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from augmentation import similarity
from augmentation.similarity import compute_matches, get_encoder


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        similarity._MODEL_CACHE.clear()

    def tearDown(self):
        similarity._MODEL_CACHE.clear()

    def test_compute_matches_returns_empty_for_empty_inputs(self):
        self.assertEqual(
            compute_matches([], np.empty((0, 2)), ["Singer.Name"], np.ones((1, 2))),
            [],
        )
        self.assertEqual(
            compute_matches(["tên"], np.ones((1, 2)), [], np.empty((0, 2))),
            [],
        )

    def test_compute_matches_keeps_every_pair_above_threshold_sorted(self):
        noun_embs = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        schema_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        matches = compute_matches(
            ["tên", "quốc gia"],
            noun_embs,
            ["Singer.Name", "Singer.Country"],
            schema_embs,
            threshold=0.75,
        )

        self.assertEqual(
            matches,
            [
                {"vi_noun": "tên", "schema_key": "Singer.Name", "similarity": 1.0},
                {"vi_noun": "quốc gia", "schema_key": "Singer.Name", "similarity": 0.8},
            ],
        )

    def test_compute_matches_discards_below_threshold(self):
        matches = compute_matches(
            ["tên"],
            np.array([[0.7, 0.0]], dtype=np.float32),
            ["Singer.Name"],
            np.array([[1.0, 0.0]], dtype=np.float32),
            threshold=0.8,
        )
        self.assertEqual(matches, [])

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


if __name__ == "__main__":
    unittest.main()
