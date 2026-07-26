import argparse
import sys
from pathlib import Path

from augmentation.pipeline import augment_examples
from augmentation.similarity import DEFAULT_E5_MODEL, cleanup_encoder
from zero_shot.common import load_dataset, load_tables, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute contextual augmentation hints for ViSpider examples"
    )
    parser.add_argument("--dataset", default="vispider", choices=["vispider"])
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    examples = load_dataset(args.dataset, args.split)
    tables = load_tables()

    if args.limit:
        print(f"[Augmentation] Limiting to first {args.limit} examples (smoke test)")
        examples = examples[: args.limit]

    hints_per_example, stats = augment_examples(
        examples=examples,
        tables=tables,
        model_name=args.model,
        threshold=args.threshold,
    )
    cleanup_encoder(args.model)

    hints_path = Path(args.output)
    stats_path = (
        Path(args.stats_output)
        if args.stats_output
        else hints_path.with_name("augment_stats.json")
    )
    output = [
        {
            "index": idx,
            "db_id": example["db_id"],
            "question_vi": example["question_vi"],
            "hints": hints,
        }
        for idx, (example, hints) in enumerate(zip(examples, hints_per_example))
    ]

    write_json(hints_path, output)
    write_json(stats_path, stats)

    examples_with_hints = stats["counts"]["examples_with_hints"]
    print(
        f"[Augmentation] {examples_with_hints}/{len(examples)} examples "
        f"have at least one hint."
    )
    print(f"[Augmentation] hints -> {hints_path}")
    print(f"[Augmentation] stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
