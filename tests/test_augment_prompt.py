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

TABLES_WITH_INTERNAL = {
    "world_1": {
        "table_names_original": ["city", "sqlite_sequence", "country"],
        "column_names_original": [
            [-1, "*"],
            [0, "Name"],
            [1, "name"],
            [1, "seq"],
            [2, "Code"],
            [2, "Name"],
        ],
        "column_types": ["text", "text", "text", "text", "text", "text"],
        "foreign_keys": [[1, 4]],
    }
}


class AugmentPromptTests(unittest.TestCase):
    def test_empty_hints_matches_baseline_prompt(self):
        self.assertEqual(
            build_prompt_augmented("Ca sĩ nào?", "db1", TABLES, []),
            build_prompt("Ca sĩ nào?", "db1", TABLES),
        )

    def test_augmented_prompt_inserts_schema_hints_before_question(self):
        prompt = build_prompt_augmented(
            "Ca sĩ ở quốc gia nào?",
            "db1",
            TABLES,
            [
                {
                    "vi_noun": "quốc gia",
                    "table": "Singer",
                    "column": "Country",
                    "similarity": 0.91,
                },
                {"vi_noun": "ca sĩ", "table": "Singer", "similarity": 0.88},
            ],
        )

        hints_pos = prompt.index("The following Vietnamese terms map to these schema items")
        question_pos = prompt.index("Question:")
        self.assertLess(hints_pos, question_pos)
        self.assertIn('  - "quốc gia" → Singer.Country', prompt)
        self.assertIn('  - "ca sĩ" → table: Singer', prompt)

    def test_prompt_omits_sqlite_internal_tables(self):
        prompt = build_prompt("Quốc gia nào?", "world_1", TABLES_WITH_INTERNAL)

        self.assertIn("Table: city", prompt)
        self.assertIn("Table: country", prompt)
        self.assertNotIn("sqlite_sequence", prompt)
        self.assertNotIn("name -> country.Code", prompt)

    def test_description_goes_before_schema_and_hints_go_after_schema(self):
        prompt = build_prompt_augmented(
            "Ca sĩ ở quốc gia nào?",
            "db1",
            TABLES,
            [{"vi_noun": "quốc gia", "table": "Singer", "similarity": 0.91}],
            description_text="Database description:\nSinger stores artist metadata.",
        )

        description_pos = prompt.index("Database description:")
        schema_pos = prompt.index("Database: db1")
        hints_pos = prompt.index("The following Vietnamese terms map to these schema items")
        question_pos = prompt.index("Question:")

        self.assertLess(description_pos, schema_pos)
        self.assertLess(schema_pos, hints_pos)
        self.assertLess(hints_pos, question_pos)


if __name__ == "__main__":
    unittest.main()
