import gc
from typing import Dict, List, Optional

import numpy as np

from augmentation.schema_nouns import (
    classify_table,
    foreign_keys_by_table,
    get_table_priority,
    is_id_column,
    is_internal_table,
)


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_EMBEDDING_MODEL = DEFAULT_E5_MODEL
E5_QUERY_PREFIX = "query: "
E5_DOCUMENT_PREFIX = "passage: "
_MODEL_CACHE = {}


def is_embeddinggemma_model(model_name: str) -> bool:
    return "embeddinggemma" in model_name.lower()


def query_prefix_for_model(model_name: str) -> str:
    return "" if is_embeddinggemma_model(model_name) else E5_QUERY_PREFIX


def document_prefix_for_model(model_name: str) -> str:
    return "" if is_embeddinggemma_model(model_name) else E5_DOCUMENT_PREFIX


def get_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def cleanup_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
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


def encode_texts(
    texts: List[str],
    encoder,
    prefix: str = "",
    batch_size: int = 128,
    task: str = "document",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if is_embeddinggemma_model(model_name):
        if task == "query":
            embeddings = encoder.encode_query(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        elif task == "document":
            embeddings = encoder.encode_document(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            raise ValueError(f"Unknown embedding task: {task}")
    else:
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
    threshold: float = 0.4,
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


def _table_columns(db_schema: Dict, table_idx: int) -> List[str]:
    return [
        column_name
        for column_table_idx, column_name in db_schema["column_names_original"]
        if column_table_idx == table_idx
    ]


def _fk_columns(db_schema: Dict) -> set:
    column_names = db_schema["column_names_original"]
    return {
        child_idx
        for child_idx, _parent_idx in db_schema.get("foreign_keys", [])
        if column_names[child_idx][0] != -1
    }


def _should_skip_column(column_name: str, col_idx: int, primary_keys: set, fk_columns: set) -> bool:
    return is_id_column(column_name) and (col_idx in primary_keys or col_idx in fk_columns)


def build_matching_targets(db_schema: Dict, db_id: str) -> List[Dict]:
    """Create table-level and column-level matching targets from tables.json."""
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    primary_keys = set(db_schema.get("primary_keys", []))
    fk_columns = _fk_columns(db_schema)
    fks_by_table = foreign_keys_by_table(db_schema)

    targets = []
    for table_idx, table_name in enumerate(table_names):
        if is_internal_table(table_name):
            continue
        columns = _table_columns(db_schema, table_idx)
        table_type = classify_table(table_name, columns, fks_by_table.get(table_idx, []))
        targets.append(
            {
                "text": table_name,
                "table": table_name,
                "column": None,
                "table_type": table_type,
            }
        )

        for col_idx, (col_table_idx, column_name) in enumerate(column_names):
            if col_table_idx != table_idx:
                continue
            if _should_skip_column(column_name, col_idx, primary_keys, fk_columns):
                continue
            targets.append(
                {
                    "text": f"{table_name} {column_name}",
                    "table": table_name,
                    "column": column_name,
                    "table_type": table_type,
                }
            )

    return targets


def _description_is_pure_pk(description: str) -> bool:
    text = (description or "").strip().lower()
    return text.startswith("pk.") and "fk" not in text


def build_rich_targets(
    schema_desc: Dict,
    db_id: str,
    db_schema: Optional[Dict] = None,
) -> List[Dict]:
    """Create semantically rich targets from schema_description_20db.json."""
    desc = schema_desc.get(db_id, {})
    if not desc:
        return build_matching_targets(db_schema, db_id) if db_schema else []

    fks_by_table = foreign_keys_by_table(db_schema) if db_schema else {}
    table_index = {
        table_name: table_idx
        for table_idx, table_name in enumerate(db_schema.get("table_names_original", []))
    } if db_schema else {}

    targets = []
    for table_name, table_info in desc.get("tables", {}).items():
        if is_internal_table(table_name):
            continue
        columns = list(table_info.get("columns", {}).keys())
        table_type = classify_table(
            table_name,
            columns,
            fks_by_table.get(table_index.get(table_name, -1), []),
        )
        table_description = table_info.get("description", "")
        table_text = " ".join(part for part in [table_name, table_description] if part)
        targets.append(
            {
                "text": table_text,
                "table": table_name,
                "column": None,
                "table_type": table_type,
            }
        )

        for column_name, column_description in table_info.get("columns", {}).items():
            if _description_is_pure_pk(column_description):
                continue
            column_text = " ".join(
                part
                for part in [table_name, column_name, column_description]
                if part
            )
            targets.append(
                {
                    "text": column_text,
                    "table": table_name,
                    "column": column_name,
                    "table_type": table_type,
                }
            )

    return targets


def match_noun_to_schema(
    vi_noun: str,
    noun_embedding: np.ndarray,
    targets: List[Dict],
    target_embs: np.ndarray,
    threshold: float = 0.8,
) -> Optional[Dict]:
    """Return the best adjusted target match for one noun candidate."""
    if not targets:
        return None

    similarities = noun_embedding @ target_embs.T
    priorities = np.array(
        [get_table_priority(target.get("table_type", "entity")) for target in targets],
        dtype=np.float32,
    )
    adjusted = similarities * priorities
    best_idx = int(np.argmax(adjusted))
    adjusted_similarity = float(adjusted[best_idx])
    if adjusted_similarity < threshold:
        return None

    target = targets[best_idx]
    return {
        "vi_noun": vi_noun,
        "table": target["table"],
        "column": target.get("column"),
        "similarity": round(adjusted_similarity, 4),
        "table_type": target.get("table_type", "entity"),
    }


def compute_target_matches(
    noun_texts: List[str],
    noun_embs: np.ndarray,
    targets: List[Dict],
    target_embs: np.ndarray,
    threshold: float = 0.8,
) -> List[Dict]:
    """Return best table/column target matches for noun candidates."""
    if not noun_texts or not targets:
        return []

    matches = []
    for noun_idx, vi_noun in enumerate(noun_texts):
        match = match_noun_to_schema(
            vi_noun,
            noun_embs[noun_idx],
            targets,
            target_embs,
            threshold=threshold,
        )
        if match is not None:
            matches.append(match)

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches
