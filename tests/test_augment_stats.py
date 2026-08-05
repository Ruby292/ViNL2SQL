import unittest

from augmentation.stats import build_augment_stats


class AugmentStatsTests(unittest.TestCase):
    def test_build_augment_stats_counts_and_unmatched_nouns(self):
        stats = build_augment_stats(
            hints_per_example=[
                [
                    {
                        "vi_noun": "tên",
                        "table": "Singer",
                        "column": "Name",
                        "similarity": 0.81,
                        "table_type": "entity",
                    },
                    {
                        "vi_noun": "quốc gia",
                        "table": "Country",
                        "column": None,
                        "similarity": 0.96,
                        "table_type": "reference",
                    },
                ],
                [],
            ],
            noun_candidates_per_example=[
                ["tên", "quốc gia", "ca sĩ"],
                ["album"],
            ],
            model_name="fake-e5",
            threshold=0.8,
            filter_stats={
                "total_nouns_before_filter": 5,
                "english_nouns_removed": 1,
                "vietnamese_stopwords_removed": 1,
                "schema_name_duplicates_removed": 0,
                "too_short_removed": 0,
                "too_long_removed": 0,
                "nouns_after_filter": 4,
                "top_removed_stopwords": {"tên": 1},
            },
        )

        self.assertEqual(stats["config"], {"model": "fake-e5", "threshold": 0.8})
        self.assertEqual(stats["counts"]["total_examples"], 2)
        self.assertEqual(stats["counts"]["examples_with_hints"], 1)
        self.assertEqual(stats["counts"]["examples_without_hints"], 1)
        self.assertEqual(stats["counts"]["avg_hints_per_example"], 1.0)
        self.assertEqual(stats["counts"]["max_hints_in_one_example"], 2)
        self.assertEqual(
            stats["similarity_distribution"]["counts"],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        )
        self.assertEqual(
            stats["top_unmatched_nouns"],
            [
                {"noun": "ca sĩ", "occurrences": 1},
                {"noun": "album", "occurrences": 1},
            ],
        )
        self.assertEqual(
            stats["filter_stats"],
            {
                "total_nouns_before_filter": 5,
                "english_nouns_removed": 1,
                "vietnamese_stopwords_removed": 1,
                "schema_name_duplicates_removed": 0,
                "too_short_removed": 0,
                "too_long_removed": 0,
                "nouns_after_filter": 4,
                "top_removed_stopwords": {"tên": 1},
            },
        )
        self.assertEqual(
            stats["table_type_distribution"],
            {"entity": 1, "reference": 1, "junction": 0},
        )
        self.assertEqual(
            stats["match_level"],
            {"table_level": 1, "column_level": 1},
        )


if __name__ == "__main__":
    unittest.main()
