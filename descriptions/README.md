# Database Description Authoring Policy

This folder stores manually written Vietnamese database descriptions for
prompt augmentation in the ViSpider Text-to-SQL pipeline.

## Output Shape

Descriptions are stored in `db_descriptions/description_db.json` as one
aggregate JSON object keyed by `db_id`.

Each database entry contains:

- `db_description`: 2-3 Vietnamese sentences about the domain and supported
  query topics.
- `tables`: one entry per non-internal table.
- `relationships`: declared and high-confidence inferred relationships.

## Table Descriptions

Each table description should say:

- what the table stores;
- when to use the table, using wording such as `Dùng khi câu hỏi hỏi về ...`;
- any data quirks, dirty values, type mismatches, typo columns, or inferred
  schema behavior.

## Table Types

Classify every table as exactly one of:

- `main`: entity or transaction table that can be queried directly.
- `lookup`: reference table used only to decode codes/IDs and not a primary
  question target.
- `junction`: many-to-many bridge table containing only foreign-key pairs and
  no direct entity attributes. Its description should say `KHÔNG chứa dữ liệu
  thực thể trực tiếp`.

If ViSpider can ask directly about the table's own data, classify it as
`main`, even if it also helps decode another table's code.

## Columns

The `columns` field is exhaustive. Every column in the schema must have a
Vietnamese description.

Use these patterns:

- Self-explanatory names: short confirmation, for example `Tên của nhân viên.`
  or `Tuổi của sinh viên.`
- Primary key/ID: `Mã định danh ...`.
- Foreign key: `Mã ..., liên kết tới parent_table.parent_col.`
- Flag/code columns: explain exact encoding when visible from samples.
- Type mismatch: note it, for example numeric values stored as text.
- Typos or ambiguous names: note the typo/ambiguity and explain from samples.
- Inferred foreign keys: state that the relation is inferred or not declared.

## Sample Values

`sample_values` should prioritize real values from SQLite:

- include 2-3 representative values for non-numeric, code, flag, categorical,
  and common filter columns;
- always include exact flag/code values, such as `T/F`, `1/0`, or type codes;
- for junction tables, include representative foreign-key pairs;
- do not fabricate values.

Numeric ID-only columns can be omitted from `sample_values` unless the format
or join behavior needs clarification.
