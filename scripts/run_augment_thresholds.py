import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augmentation.pipeline import augment_examples
from augmentation.pos_filter import extract_noun_candidates, filter_nouns_with_stats
from augmentation.schema_nouns import schema_name_set
from augmentation.similarity import DEFAULT_EMBEDDING_MODEL, cleanup_encoder
from augmentation.stats import build_augment_stats
from zero_shot.common import load_dataset, load_tables, write_json


DEFAULT_THRESHOLDS = [0.82, 0.85, 0.87, 0.90]
DEFAULT_SCHEMA_DESC = (
    REPO_ROOT / "descriptions" / "db_descriptions" / "schema_description_20db.json"
)


def threshold_tag(value: float) -> str:
    return f"t{value:.2f}".replace(".", "")


def repo_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run augmentation once at the lowest threshold and materialize "
            "hints/stats for multiple higher thresholds."
        )
    )
    parser.add_argument("--dataset", default="vispider", choices=["vispider"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--schema-desc",
        default=str(DEFAULT_SCHEMA_DESC),
        help="Path to schema_description_20db.json. Use --no-schema-desc to disable.",
    )
    parser.add_argument("--no-schema-desc", action="store_true")
    parser.add_argument("--output-root", default="augmentation/result_new_aug")
    parser.add_argument(
        "--output-suffix",
        default="_e5",
        help="Suffix appended to threshold directories, for example dev_t085_e5.",
    )
    return parser.parse_args()


def load_schema_desc(args) -> dict | None:
    if args.no_schema_desc:
        return None

    schema_desc_path = repo_relative_path(args.schema_desc)
    if not schema_desc_path.exists():
        raise FileNotFoundError(f"Schema description file not found: {schema_desc_path}")

    with open(schema_desc_path, "r", encoding="utf-8") as handle:
        schema_desc = json.load(handle)
    print(f"[Augmentation] Loaded schema descriptions from {schema_desc_path}")
    return schema_desc


def collect_filtered_nouns(examples, tables):
    schema_names_by_db = {
        db_id: schema_name_set(tables[db_id])
        for db_id in {example["db_id"] for example in examples}
    }
    filter_stats = Counter(
        {
            "total_nouns_before_filter": 0,
            "english_nouns_removed": 0,
            "vietnamese_stopwords_removed": 0,
            "schema_name_duplicates_removed": 0,
            "too_short_removed": 0,
            "too_long_removed": 0,
            "nouns_after_filter": 0,
        }
    )
    removed_stopwords = Counter()
    noun_candidates_per_example = []

    for example in examples:
        raw_candidates = extract_noun_candidates(example["question_vi"])
        filtered, stats = filter_nouns_with_stats(
            raw_candidates,
            schema_names_by_db[example["db_id"]],
        )
        noun_candidates_per_example.append(filtered)
        for key, value in stats.items():
            if key == "top_removed_stopwords":
                removed_stopwords.update(value)
            else:
                filter_stats[key] += value

    aggregate_filter_stats = dict(filter_stats)
    aggregate_filter_stats["top_removed_stopwords"] = dict(
        removed_stopwords.most_common(10)
    )
    return noun_candidates_per_example, aggregate_filter_stats


def materialize_threshold(
    examples,
    base_hints_per_example,
    noun_candidates_per_example,
    filter_stats,
    threshold,
    model_name,
    output_root,
    output_suffix,
    split,
):
    hints_per_example = [
        [
            hint
            for hint in hints
            if float(hint["similarity"]) >= threshold
        ]
        for hints in base_hints_per_example
    ]
    stats = build_augment_stats(
        hints_per_example=hints_per_example,
        noun_candidates_per_example=noun_candidates_per_example,
        model_name=model_name,
        threshold=threshold,
        filter_stats=filter_stats,
    )
    output_items = [
        {
            "index": idx,
            "db_id": example["db_id"],
            "question_vi": example["question_vi"],
            "hints": hints,
        }
        for idx, (example, hints) in enumerate(zip(examples, hints_per_example))
    ]

    out_dir = output_root / f"{split}_{threshold_tag(threshold)}{output_suffix}"
    hints_path = out_dir / "hints.json"
    stats_path = out_dir / "augment_stats.json"
    write_json(hints_path, output_items)
    write_json(stats_path, stats)

    counts = stats["counts"]
    match_level = stats["match_level"]
    print(
        f"[Hints] threshold={threshold:.2f} -> {hints_path} "
        f"({counts['examples_with_hints']}/{counts['total_examples']} examples, "
        f"avg={counts['avg_hints_per_example']:.2f}, "
        f"table={match_level['table_level']}, column={match_level['column_level']})"
    )
    print(f"[Hints] stats -> {stats_path}")


def main() -> int:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds:
        raise ValueError("At least one threshold is required")

    examples = load_dataset(args.dataset, args.split)
    if args.limit:
        print(f"[Augmentation] Limiting to first {args.limit} examples (smoke test)")
        examples = examples[: args.limit]
    tables = load_tables()
    schema_desc = load_schema_desc(args)

    base_threshold = min(thresholds)
    print(
        f"[Augmentation] Running base augmentation at threshold={base_threshold:.2f} "
        f"for {len(examples)} examples..."
    )
    try:
        base_hints_per_example, _base_stats = augment_examples(
            examples=examples,
            tables=tables,
            model_name=args.model,
            threshold=base_threshold,
            schema_desc=schema_desc,
        )
    finally:
        cleanup_encoder(args.model)

    print("[Augmentation] Rebuilding filtered noun stats for materialized thresholds...")
    noun_candidates_per_example, filter_stats = collect_filtered_nouns(examples, tables)

    output_root = repo_relative_path(args.output_root)
    for threshold in thresholds:
        materialize_threshold(
            examples=examples,
            base_hints_per_example=base_hints_per_example,
            noun_candidates_per_example=noun_candidates_per_example,
            filter_stats=filter_stats,
            threshold=threshold,
            model_name=args.model,
            output_root=output_root,
            output_suffix=args.output_suffix,
            split=args.split,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
