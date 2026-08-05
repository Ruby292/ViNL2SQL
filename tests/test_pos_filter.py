import sys
import types
import unittest
from unittest.mock import patch

from augmentation.pos_filter import (
    extract_noun_candidates,
    extract_nouns,
    filter_nouns,
    filter_nouns_with_stats,
    is_stopword,
    is_vietnamese_noun,
)


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

    def test_filter_nouns_removes_english_schema_names_and_bad_lengths(self):
        nouns = ["concert", "ca sĩ", "x", "tên ca sĩ nổi tiếng nhất trong toàn bộ hệ thống"]
        kept, stats = filter_nouns_with_stats(
            nouns,
            {"concert", "singer_id"},
        )

        self.assertEqual(kept, ["ca sĩ"])
        self.assertEqual(stats["total_nouns_before_filter"], 4)
        self.assertEqual(stats["schema_name_duplicates_removed"], 1)
        self.assertEqual(stats["too_short_removed"], 1)
        self.assertEqual(stats["too_long_removed"], 1)
        self.assertEqual(stats["vietnamese_stopwords_removed"], 0)
        self.assertEqual(stats["nouns_after_filter"], 1)
        self.assertEqual(stats["top_removed_stopwords"], {})

    def test_filter_nouns_removes_ascii_heavy_english_terms(self):
        self.assertEqual(filter_nouns(["template", "quốc gia"], set()), ["quốc gia"])
        self.assertFalse(is_vietnamese_noun("stadium"))
        self.assertTrue(is_vietnamese_noun("sân vận động"))

    def test_filter_nouns_removes_vietnamese_stopwords_case_insensitively(self):
        nouns = [
            "tên",
            "Tên",
            "số lượng",
            "Tổng số",
            "Hiển thị",
            "Liệt kê",
            "ca sĩ",
        ]

        kept, stats = filter_nouns_with_stats(nouns, set())

        self.assertTrue(is_stopword("Tên"))
        self.assertTrue(is_stopword(" tổng số "))
        self.assertFalse(is_stopword("ca sĩ"))
        self.assertFalse(is_vietnamese_noun("tên"))
        self.assertEqual(kept, ["ca sĩ"])
        self.assertEqual(stats["vietnamese_stopwords_removed"], 6)
        self.assertEqual(stats["top_removed_stopwords"]["tên"], 2)
        self.assertEqual(stats["top_removed_stopwords"]["số lượng"], 1)


if __name__ == "__main__":
    unittest.main()
