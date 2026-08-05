import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augmentation.pos_filter import extract_noun_candidates, filter_nouns_with_stats
from augmentation.schema_nouns import schema_name_set
from augmentation.stats import build_augment_stats
from zero_shot.common import load_dataset, load_tables, write_json


def threshold_tag(value: float) -> str:
    return f"t{value:.2f}".replace(".", "")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-threshold augmentation hints from a lower-threshold hints file."
    )
    parser.add_argument("--dataset", default="vispider", choices=["vispider"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--source-hints", required=True)
    parser.add_argument("--source-stats")
    parser.add_argument("--output-root", default="augmentation/results_embeddinggemma")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to threshold directories, for example _e5.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45])
    parser.add_argument("--model-name")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    source_hints_path = Path(args.source_hints)
    source_hints = load_json(source_hints_path)

    source_stats = {}
    if args.source_stats:
        source_stats_path = Path(args.source_stats)
        if source_stats_path.exists():
            source_stats = load_json(source_stats_path)

    model_name = (
        args.model_name
        or source_stats.get("config", {}).get("model")
        or "unknown"
    )

    examples = load_dataset(args.dataset, args.split)
    if len(source_hints) != len(examples):
        raise ValueError(
            f"Source hints count ({len(source_hints)}) != example count ({len(examples)})"
        )

    tables = load_tables()
    noun_candidates_per_example, filter_stats = collect_filtered_nouns(examples, tables)

    output_root = Path(args.output_root)
    for threshold in args.thresholds:
        tag = threshold_tag(threshold)
        out_dir = output_root / f"{args.split}_{tag}{args.output_suffix}"

        output_items = []
        hints_per_example = []
        for item in source_hints:
            hints = [
                hint
                for hint in item["hints"]
                if float(hint["similarity"]) >= threshold
            ]
            hints_per_example.append(hints)
            output_items.append(
                {
                    "index": item["index"],
                    "db_id": item["db_id"],
                    "question_vi": item["question_vi"],
                    "hints": hints,
                }
            )

        stats = build_augment_stats(
            hints_per_example=hints_per_example,
            noun_candidates_per_example=noun_candidates_per_example,
            model_name=model_name,
            threshold=threshold,
            filter_stats=filter_stats,
        )

        hints_path = out_dir / "hints.json"
        stats_path = out_dir / "augment_stats.json"
        write_json(hints_path, output_items)
        write_json(stats_path, stats)

        counts = stats["counts"]
        print(
            f"[Hints] threshold={threshold:.2f} -> {hints_path} "
            f"({counts['examples_with_hints']}/{counts['total_examples']} examples, "
            f"avg={counts['avg_hints_per_example']:.2f})"
        )
        print(f"[Hints] stats -> {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
