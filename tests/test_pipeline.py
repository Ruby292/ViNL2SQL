import unittest

from augmentation.pipeline import dedupe_and_limit_hints


class PipelineTests(unittest.TestCase):
    def test_dedupe_and_limit_hints_keeps_best_per_table_and_top_three(self):
        hints = [
            {"vi_noun": "ca sĩ", "table": "Singer", "similarity": 0.83},
            {"vi_noun": "tên ca sĩ", "table": "Singer", "similarity": 0.91},
            {"vi_noun": "buổi hòa nhạc", "table": "Concert", "similarity": 0.88},
            {"vi_noun": "sân vận động", "table": "Stadium", "similarity": 0.86},
            {"vi_noun": "bài hát", "table": "Song", "similarity": 0.85},
        ]

        self.assertEqual(
            dedupe_and_limit_hints(hints),
            [
                {"vi_noun": "tên ca sĩ", "table": "Singer", "similarity": 0.91},
                {"vi_noun": "buổi hòa nhạc", "table": "Concert", "similarity": 0.88},
                {"vi_noun": "sân vận động", "table": "Stadium", "similarity": 0.86},
            ],
        )


if __name__ == "__main__":
    unittest.main()
