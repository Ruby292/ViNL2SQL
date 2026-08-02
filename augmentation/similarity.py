import gc
from typing import Dict, List

import numpy as np


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-large-instruct"
E5_QUERY_INSTRUCTION = (
    "Instruct: Given a Vietnamese noun from a natural language question, "
    "retrieve the matching database table or column identifier.\n"
    "Query: "
)
_MODEL_CACHE = {}


def get_encoder(model_name: str = DEFAULT_E5_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def cleanup_encoder(model_name: str = DEFAULT_E5_MODEL) -> None:
    encoder = _MODEL_CACHE.pop(model_name, None)
    if encoder is not None:
        del encoder
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_texts(texts: List[str], encoder, prefix: str = "", batch_size: int = 128) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = encoder.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def compute_matches(
    noun_x_texts: List[str],
    noun_x_embs: np.ndarray,
    table_names: List[str],
    table_embs: np.ndarray,
    threshold: float = 0.8,
) -> List[Dict]:
    """Return the best table match for each noun candidate above threshold."""
    if not noun_x_texts or not table_names:
        return []

    sim_matrix = noun_x_embs @ table_embs.T
    matches = []
    for noun_idx, vi_noun in enumerate(noun_x_texts):
        best_table_idx = int(np.argmax(sim_matrix[noun_idx]))
        similarity = float(sim_matrix[noun_idx, best_table_idx])
        if similarity >= threshold:
            matches.append(
                {
                    "vi_noun": vi_noun,
                    "table": table_names[best_table_idx],
                    "similarity": round(similarity, 4),
                }
            )

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches
