import unittest

from augmentation.schema_nouns import clean_identifier, extract_schema_nouns


class SchemaNounsTests(unittest.TestCase):
    def test_clean_identifier_handles_underscores_and_camel_case(self):
        self.assertEqual(clean_identifier("Birth_Date"), "birth date")
        self.assertEqual(clean_identifier("CustomerName"), "customer name")
        self.assertEqual(clean_identifier("People_ID"), "people id")

    def test_extract_schema_nouns_uses_original_table_and_qualified_columns(self):
        schema = {
            "table_names_original": ["Singer", "Concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [1, "Concert_ID"],
            ],
        }

        nouns = extract_schema_nouns(schema)

        self.assertEqual(nouns["Singer"], "singer")
        self.assertEqual(nouns["Concert"], "concert")
        self.assertEqual(nouns["Singer.Singer_ID"], "singer singer id")
        self.assertEqual(nouns["Singer.Name"], "singer name")
        self.assertEqual(nouns["Concert.Concert_ID"], "concert concert id")
        self.assertNotIn("Singer.*", nouns)


if __name__ == "__main__":
    unittest.main()
