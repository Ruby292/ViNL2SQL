import unittest

from augmentation.schema_nouns import clean_identifier, extract_schema_nouns, is_id_column


class SchemaNounsTests(unittest.TestCase):
    def test_clean_identifier_handles_underscores_and_camel_case(self):
        self.assertEqual(clean_identifier("Birth_Date"), "birth date")
        self.assertEqual(clean_identifier("CustomerName"), "customer name")
        self.assertEqual(clean_identifier("People_ID"), "people id")

    def test_is_id_column_handles_exact_id_and_id_suffix(self):
        self.assertTrue(is_id_column("id"))
        self.assertTrue(is_id_column("ID"))
        self.assertTrue(is_id_column("Singer_ID"))
        self.assertTrue(is_id_column("singer_id"))
        self.assertFalse(is_id_column("Song_Name"))

    def test_extract_schema_nouns_builds_table_level_text_without_id_columns(self):
        schema = {
            "table_names_original": ["Singer", "Concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [0, "Country"],
                [0, "Song_Name"],
                [1, "Concert_ID"],
                [1, "Venue"],
            ],
        }

        nouns = extract_schema_nouns(schema)

        self.assertEqual(nouns["Singer"], "Singer: Name, Country, Song_Name")
        self.assertEqual(nouns["Concert"], "Concert: Venue")
        self.assertNotIn("Singer.Singer_ID", nouns)
        self.assertNotIn("Concert.Concert_ID", nouns)
        self.assertNotIn("Singer.*", nouns)


if __name__ == "__main__":
    unittest.main()
