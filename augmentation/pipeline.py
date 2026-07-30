from typing import Dict, List, Tuple

from augmentation.pos_filter import extract_noun_candidates
from augmentation.schema_nouns import extract_schema_nouns
from augmentation.similarity import (
    DEFAULT_E5_MODEL,
    E5_QUERY_INSTRUCTION,
    compute_matches,
    encode_texts,
    get_encoder,
)
from augmentation.stats import build_augment_stats


def augment_examples(
    examples: List[Dict],
    tables: Dict,
    model_name: str = DEFAULT_E5_MODEL,
    threshold: float = 0.8,
) -> Tuple[List[List[Dict]], Dict]:
    print(f"[Augmentation] Loading encoder: {model_name}")
    encoder = get_encoder(model_name)

    print(f"[Augmentation] Extracting noun candidates from {len(examples)} questions...")
    noun_candidates_per_example = [
        extract_noun_candidates(example["question_vi"])
        for example in examples
    ]

    db_ids = sorted({example["db_id"] for example in examples})
    schema_embedding_cache = {}
    print(f"[Augmentation] Encoding schema candidates for {len(db_ids)} databases...")
    for db_id in db_ids:
        schema_noun_map = extract_schema_nouns(tables[db_id])
        schema_keys = list(schema_noun_map.keys())
        schema_texts = list(schema_noun_map.values())
        schema_embs = encode_texts(schema_texts, encoder)
        schema_embedding_cache[db_id] = (schema_keys, schema_embs)

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
            prefix=E5_QUERY_INSTRUCTION,
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

        schema_keys, schema_embs = schema_embedding_cache[example["db_id"]]
        candidate_indexes = [candidate_to_idx[candidate] for candidate in candidates]
        hints = compute_matches(
            noun_x_texts=candidates,
            noun_x_embs=unique_embs[candidate_indexes],
            schema_keys=schema_keys,
            schema_embs=schema_embs,
            threshold=threshold,
        )
        hints_per_example.append(hints)

    stats = build_augment_stats(
        hints_per_example=hints_per_example,
        noun_candidates_per_example=noun_candidates_per_example,
        model_name=model_name,
        threshold=threshold,
    )
    return hints_per_example, stats
