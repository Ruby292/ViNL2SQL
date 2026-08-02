import unittest

from zero_shot.prompts import build_prompt, build_prompt_augmented


TABLES = {
    "db1": {
        "table_names_original": ["Singer"],
        "column_names_original": [[-1, "*"], [0, "Name"], [0, "Country"]],
        "column_types": ["text", "text", "text"],
        "foreign_keys": [],
    }
}


class AugmentPromptTests(unittest.TestCase):
    def test_empty_hints_matches_baseline_prompt(self):
        self.assertEqual(
            build_prompt_augmented("Ca sĩ nào?", "db1", TABLES, []),
            build_prompt("Ca sĩ nào?", "db1", TABLES),
        )

    def test_augmented_prompt_inserts_table_hints_before_question(self):
        prompt = build_prompt_augmented(
            "Ca sĩ ở quốc gia nào?",
            "db1",
            TABLES,
            [{"vi_noun": "quốc gia", "table": "Singer", "similarity": 0.91}],
        )

        hints_pos = prompt.index("The following Vietnamese terms map to these tables")
        question_pos = prompt.index("Question:")
        self.assertLess(hints_pos, question_pos)
        self.assertIn('  - "quốc gia" → table: Singer', prompt)


if __name__ == "__main__":
    unittest.main()
