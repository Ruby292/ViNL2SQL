import unittest

from augmentation.schema_nouns import (
    classify_table,
    clean_identifier,
    extract_schema_nouns,
    foreign_keys_by_table,
    get_table_priority,
    is_id_column,
    is_internal_table,
    schema_name_set,
)


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

    def test_is_internal_table_handles_sqlite_prefix(self):
        self.assertTrue(is_internal_table("sqlite_sequence"))
        self.assertTrue(is_internal_table(" SQLITE_SEQUENCE "))
        self.assertTrue(is_internal_table("sqlite_stat1"))
        self.assertFalse(is_internal_table("country"))

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

    def test_extract_schema_nouns_skips_sqlite_internal_tables(self):
        schema = {
            "table_names_original": ["city", "sqlite_sequence", "country"],
            "column_names_original": [
                [-1, "*"],
                [0, "Name"],
                [1, "name"],
                [1, "seq"],
                [2, "Code"],
                [2, "Name"],
            ],
        }

        nouns = extract_schema_nouns(schema)

        self.assertNotIn("sqlite_sequence", nouns)
        self.assertEqual(nouns["city"], "city: Name")
        self.assertEqual(nouns["country"], "country: Code, Name")

    def test_classify_table_identifies_reference_tables(self):
        self.assertEqual(
            classify_table(
                "Ref_Template_Types",
                ["Template_Type_Code", "Template_Type_Description"],
                [],
            ),
            "reference",
        )
        self.assertEqual(
            classify_table("status_codes", ["code", "description"], []),
            "reference",
        )

    def test_classify_table_identifies_junction_and_entity_tables(self):
        foreign_keys = [
            {"column": "Singer_ID", "ref_table": "singer"},
            {"column": "Concert_ID", "ref_table": "concert"},
        ]

        self.assertEqual(
            classify_table(
                "singer_in_concert",
                ["Singer_ID", "Concert_ID"],
                foreign_keys,
            ),
            "junction",
        )
        self.assertEqual(
            classify_table("singer", ["Singer_ID", "Name", "Country"], []),
            "entity",
        )

    def test_get_table_priority_prefers_entity_targets(self):
        self.assertEqual(get_table_priority("entity"), 1.0)
        self.assertEqual(get_table_priority("junction"), 0.7)
        self.assertEqual(get_table_priority("reference"), 0.6)

    def test_schema_name_set_uses_non_internal_tables_and_columns(self):
        schema = {
            "table_names_original": ["Singer", "sqlite_sequence"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [1, "seq"],
            ],
        }

        self.assertEqual(schema_name_set(schema), {"singer", "singer_id", "name"})

    def test_foreign_keys_by_table_groups_child_fk_columns(self):
        schema = {
            "table_names_original": ["singer", "concert", "singer_in_concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [1, "Concert_ID"],
                [2, "Singer_ID"],
                [2, "Concert_ID"],
            ],
            "foreign_keys": [[3, 1], [4, 2]],
        }

        grouped = foreign_keys_by_table(schema)

        self.assertEqual(grouped[0], [])
        self.assertEqual(
            grouped[2],
            [
                {
                    "column": "Singer_ID",
                    "ref_table": "singer",
                    "ref_column": "Singer_ID",
                },
                {
                    "column": "Concert_ID",
                    "ref_table": "concert",
                    "ref_column": "Concert_ID",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
