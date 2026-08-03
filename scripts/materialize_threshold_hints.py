import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augmentation.pos_filter import extract_noun_candidates
from augmentation.stats import build_augment_stats
from zero_shot.common import load_dataset, write_json


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
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.45])
    parser.add_argument("--model-name")
    return parser.parse_args()


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

    noun_candidates_per_example = [
        extract_noun_candidates(example["question_vi"])
        for example in examples
    ]

    output_root = Path(args.output_root)
    for threshold in args.thresholds:
        tag = threshold_tag(threshold)
        out_dir = output_root / f"{args.split}_{tag}"

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
