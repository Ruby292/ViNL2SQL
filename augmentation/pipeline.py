from collections import Counter
from typing import Dict, List, Optional, Tuple

from augmentation.pos_filter import extract_noun_candidates, filter_nouns_with_stats
from augmentation.schema_nouns import schema_name_set
from augmentation.similarity import (
    DEFAULT_EMBEDDING_MODEL,
    build_matching_targets,
    build_rich_targets,
    compute_target_matches,
    document_prefix_for_model,
    encode_texts,
    get_encoder,
    query_prefix_for_model,
)
from augmentation.stats import build_augment_stats


MAX_HINTS_PER_QUESTION = 3


def dedupe_and_limit_hints(
    hints: List[Dict],
    max_hints: int = MAX_HINTS_PER_QUESTION,
) -> List[Dict]:
    """Keep the strongest noun match for each schema target and cap hints."""
    best_by_target: Dict[Tuple[str, str], Dict] = {}
    for hint in hints:
        target = (hint["table"], hint.get("column") or "")
        current = best_by_target.get(target)
        if current is None or hint["similarity"] > current["similarity"]:
            best_by_target[target] = hint

    deduped = sorted(
        best_by_target.values(),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return deduped[:max_hints]


def augment_examples(
    examples: List[Dict],
    tables: Dict,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    threshold: float = 0.4,
    schema_desc: Optional[Dict] = None,
) -> Tuple[List[List[Dict]], Dict]:
    """Create table/column-level schema hints for ViSpider examples."""
    print(f"[Augmentation] Loading encoder: {model_name}")
    encoder = get_encoder(model_name)

    print(f"[Augmentation] Extracting noun candidates from {len(examples)} questions...")
    raw_noun_candidates_per_example = [
        extract_noun_candidates(example["question_vi"])
        for example in examples
    ]
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
    for example, candidates in zip(examples, raw_noun_candidates_per_example):
        filtered, stats = filter_nouns_with_stats(
            candidates,
            schema_names_by_db[example["db_id"]],
        )
        noun_candidates_per_example.append(filtered)
        for key, value in stats.items():
            if key == "top_removed_stopwords":
                removed_stopwords.update(value)
            else:
                filter_stats[key] += value

    db_ids = sorted({example["db_id"] for example in examples})
    schema_embedding_cache = {}
    target_mode = "rich schema-description" if schema_desc else "schema identifier"
    print(f"[Augmentation] Encoding {target_mode} targets for {len(db_ids)} databases...")
    for db_id in db_ids:
        targets = (
            build_rich_targets(schema_desc, db_id, tables[db_id])
            if schema_desc
            else build_matching_targets(tables[db_id], db_id)
        )
        target_texts = [target["text"] for target in targets]
        target_embs = encode_texts(
            target_texts,
            encoder,
            prefix=document_prefix_for_model(model_name),
            task="document",
            model_name=model_name,
        )
        schema_embedding_cache[db_id] = (targets, target_embs)

    flat_candidates = []
    for candidates in noun_candidates_per_example:
        flat_candidates.extend(candidates)

    if flat_candidates:
        unique_candidates = list(dict.fromkeys(flat_candidates))
        print(
            f"[Augmentation] Encoding {len(unique_candidates)} unique question "
            f"candidates ({len(flat_candidates)} total)..."
        )
        unique_embs = encode_texts(
            unique_candidates,
            encoder,
            prefix=query_prefix_for_model(model_name),
            task="query",
            model_name=model_name,
        )
        candidate_to_idx = {
            candidate: idx
            for idx, candidate in enumerate(unique_candidates)
        }
    else:
        unique_embs = None
        candidate_to_idx = {}

    hints_per_example = []
    for idx, (example, candidates) in enumerate(zip(examples, noun_candidates_per_example)):
        if not candidates:
            hints_per_example.append([])
            continue

        targets, target_embs = schema_embedding_cache[example["db_id"]]
        candidate_indexes = [candidate_to_idx[candidate] for candidate in candidates]
        raw_hints = compute_target_matches(
            noun_texts=candidates,
            noun_embs=unique_embs[candidate_indexes],
            targets=targets,
            target_embs=target_embs,
            threshold=threshold,
        )
        hints = dedupe_and_limit_hints(raw_hints)
        hints_per_example.append(hints)

    aggregate_filter_stats = dict(filter_stats)
    aggregate_filter_stats["top_removed_stopwords"] = dict(
        removed_stopwords.most_common(10)
    )

    stats = build_augment_stats(
        hints_per_example=hints_per_example,
        noun_candidates_per_example=noun_candidates_per_example,
        model_name=model_name,
        threshold=threshold,
        filter_stats=aggregate_filter_stats,
    )
    return hints_per_example, stats
