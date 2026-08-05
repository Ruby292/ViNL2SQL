from collections import Counter
from typing import Dict, List, Sequence


SIMILARITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
FILTER_STAT_KEYS = [
    "total_nouns_before_filter",
    "english_nouns_removed",
    "vietnamese_stopwords_removed",
    "schema_name_duplicates_removed",
    "too_short_removed",
    "too_long_removed",
    "nouns_after_filter",
]


def _default_filter_stats(noun_candidates_per_example: Sequence[Sequence[str]]) -> Dict:
    total_candidates = sum(len(candidates) for candidates in noun_candidates_per_example)
    return {
        "total_nouns_before_filter": total_candidates,
        "english_nouns_removed": 0,
        "vietnamese_stopwords_removed": 0,
        "schema_name_duplicates_removed": 0,
        "too_short_removed": 0,
        "too_long_removed": 0,
        "nouns_after_filter": total_candidates,
        "top_removed_stopwords": {},
    }


def _normalize_filter_stats(
    filter_stats: Dict,
    noun_candidates_per_example: Sequence[Sequence[str]],
) -> Dict:
    normalized = _default_filter_stats(noun_candidates_per_example)
    if not filter_stats:
        return normalized

    for key in FILTER_STAT_KEYS:
        normalized[key] = filter_stats.get(key, normalized[key])
    normalized["top_removed_stopwords"] = filter_stats.get("top_removed_stopwords", {})
    return normalized


def build_augment_stats(
    hints_per_example: Sequence[Sequence[Dict]],
    noun_candidates_per_example: Sequence[Sequence[str]],
    model_name: str,
    threshold: float,
    filter_stats: Dict = None,
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

    table_type_counter = Counter(
        hint.get("table_type", "unknown")
        for hints in hints_per_example
        for hint in hints
    )
    table_type_distribution = {
        "entity": table_type_counter.get("entity", 0),
        "reference": table_type_counter.get("reference", 0),
        "junction": table_type_counter.get("junction", 0),
    }
    for table_type, count in table_type_counter.items():
        if table_type not in table_type_distribution:
            table_type_distribution[table_type] = count

    match_level = Counter(
        "column_level" if hint.get("column") else "table_level"
        for hints in hints_per_example
        for hint in hints
    )

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
        "filter_stats": _normalize_filter_stats(
            filter_stats,
            noun_candidates_per_example,
        ),
        "table_type_distribution": table_type_distribution,
        "match_level": {
            "table_level": match_level.get("table_level", 0),
            "column_level": match_level.get("column_level", 0),
        },
    }
