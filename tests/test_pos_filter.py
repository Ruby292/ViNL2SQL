import sys
import types
import unittest
from unittest.mock import patch

from augmentation.pos_filter import extract_noun_candidates, extract_nouns


class PosFilterTests(unittest.TestCase):
    def test_extract_nouns_keeps_only_noun_tags(self):
        fake_underthesea = types.SimpleNamespace(
            pos_tag=lambda text: [("sinh", "V"), ("viên", "N"), ("Hà Nội", "Np")]
        )
        with patch.dict(sys.modules, {"underthesea": fake_underthesea}):
            self.assertEqual(extract_nouns("ignored"), ["viên", "Hà Nội"])

    def test_extract_noun_candidates_merges_consecutive_nouns(self):
        fake_underthesea = types.SimpleNamespace(
            pos_tag=lambda text: [
                ("cơ", "N"),
                ("sở", "N"),
                ("dữ", "N"),
                ("liệu", "N"),
                ("nào", "P"),
            ]
        )
        with patch.dict(sys.modules, {"underthesea": fake_underthesea}):
            self.assertEqual(
                extract_noun_candidates("ignored"),
                ["cơ", "sở", "dữ", "liệu", "cơ sở dữ liệu"],
            )

    def test_extract_noun_candidates_deduplicates_in_order(self):
        fake_underthesea = types.SimpleNamespace(
            pos_tag=lambda text: [("bảng", "N"), ("bảng", "N"), ("cột", "N")]
        )
        with patch.dict(sys.modules, {"underthesea": fake_underthesea}):
            self.assertEqual(
                extract_noun_candidates("ignored"),
                ["bảng", "cột", "bảng bảng cột"],
            )

    def test_module_import_is_lazy(self):
        self.assertIn("augmentation.pos_filter", sys.modules)


if __name__ == "__main__":
    unittest.main()
