from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

BGE_MODEL_NAME = "BAAI/bge-m3"

_bge_model: SentenceTransformer | None = None


def get_bge_model() -> SentenceTransformer:
    global _bge_model
    if _bge_model is None:
        _bge_model = SentenceTransformer(BGE_MODEL_NAME)
    return _bge_model


def build_dense_index(
    chunks: List[str],
    model: SentenceTransformer,
    normalize: bool = True,
    show_progress: bool = False,
    batch_size: int = 8,
) -> np.ndarray:
    embs = model.encode(chunks, batch_size=batch_size, show_progress_bar=show_progress)
    embs = np.asarray(embs, dtype="float32")
    if normalize and len(embs) > 0:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    return embs


def dense_retrieve_with_index(
    query: str,
    index: Dict,
    top_k: int = 6,
    normalize_query: bool = True,
    prefix_query: bool = True,
) -> List[Tuple[int, float]]:
    chunks = index["chunks"]
    doc_embs = index.get("dense_embeddings")
    model = index.get("dense_model")

    if (not chunks) or (doc_embs is None) or (model is None):
        return []

    qtext = f"query: {query}" if prefix_query else query
    q_emb = model.encode([qtext])[0].astype("float32")
    if normalize_query:
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    sims = np.dot(doc_embs, q_emb)
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idx]
