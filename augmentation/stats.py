from collections import Counter
from typing import Dict, List, Sequence


SIMILARITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]


def build_augment_stats(
    hints_per_example: Sequence[Sequence[Dict]],
    noun_candidates_per_example: Sequence[Sequence[str]],
    model_name: str,
    threshold: float,
) -> Dict:
    total = len(hints_per_example)
    hint_counts = [len(hints) for hints in hints_per_example]
    examples_with_hints = sum(1 for count in hint_counts if count > 0)
    similarities = [
        hint["similarity"]
        for hints in hints_per_example
        for hint in hints
    ]

    bin_counts = []
    for lower, upper in zip(SIMILARITY_BINS, SIMILARITY_BINS[1:]):
        if upper == SIMILARITY_BINS[-1]:
            count = sum(1 for score in similarities if lower <= score <= upper)
        else:
            count = sum(1 for score in similarities if lower <= score < upper)
        bin_counts.append(count)

    unmatched = Counter()
    for hints, candidates in zip(hints_per_example, noun_candidates_per_example):
        matched_nouns = {hint["vi_noun"] for hint in hints}
        for candidate in candidates:
            if candidate not in matched_nouns:
                unmatched[candidate] += 1

    return {
        "config": {
            "model": model_name,
            "threshold": threshold,
        },
        "counts": {
            "total_examples": total,
            "examples_with_hints": examples_with_hints,
            "examples_without_hints": total - examples_with_hints,
            "avg_hints_per_example": (sum(hint_counts) / total) if total else 0.0,
            "max_hints_in_one_example": max(hint_counts) if hint_counts else 0,
        },
        "similarity_distribution": {
            "bins": SIMILARITY_BINS,
            "counts": bin_counts,
        },
        "top_unmatched_nouns": [
            {"noun": noun, "occurrences": count}
            for noun, count in unmatched.most_common(10)
        ],
    }
