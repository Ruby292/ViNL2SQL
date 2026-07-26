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
    offsets = []
    for candidates in noun_candidates_per_example:
        offsets.append(len(flat_candidates))
        flat_candidates.extend(candidates)

    if flat_candidates:
        print(f"[Augmentation] Encoding {len(flat_candidates)} question candidates...")
        flat_embs = encode_texts(flat_candidates, encoder, prefix=E5_QUERY_INSTRUCTION)
    else:
        flat_embs = None

    hints_per_example = []
    for idx, (example, candidates) in enumerate(zip(examples, noun_candidates_per_example)):
        if not candidates:
            hints_per_example.append([])
            continue

        start = offsets[idx]
        end = start + len(candidates)
        schema_keys, schema_embs = schema_embedding_cache[example["db_id"]]
        hints = compute_matches(
            noun_x_texts=candidates,
            noun_x_embs=flat_embs[start:end],
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
