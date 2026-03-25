import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import build_dense_index, get_bge_model
from .io_utils import read_file

INDEX_VERSION = 1


# ── Chunking ──────────────────────────────────────────────────────────────────

def page_level_chunking(text: str) -> List[str]:
    parts = text.split("# Page")[1:]
    return [f"# Page{p}" for p in parts]


# ── Persistence helpers ───────────────────────────────────────────────────────

def make_index_cache_path(doc_path: str, index_root: str) -> str:
    os.makedirs(index_root, exist_ok=True)
    h = hashlib.sha256(doc_path.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(doc_path)
    return os.path.join(index_root, f"{base}.{h}.pkl")


def serialize_index(index: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": INDEX_VERSION,
        "doc_id": index["doc_id"],
        "doc_path": index["doc_path"],
        "chunks": index["chunks"],
        "dense_embeddings": index["dense_embeddings"],
    }


def deserialize_index(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("version", 0) != INDEX_VERSION:
        print(f"[WARN] Index version mismatch (found {data.get('version')}, expected {INDEX_VERSION}).")
    return {
        "doc_id": data["doc_id"],
        "doc_path": data["doc_path"],
        "chunks": data["chunks"],
        "dense_embeddings": data["dense_embeddings"],
        "dense_model": get_bge_model(),
    }


def save_doc_index_to_disk(index: Dict[str, Any], index_root: str) -> str:
    cache_path = make_index_cache_path(index["doc_path"], index_root)
    with open(cache_path, "wb") as f:
        pickle.dump(serialize_index(index), f)
    return cache_path


def load_doc_index_from_disk(doc_path: str, index_root: str) -> Optional[Dict[str, Any]]:
    cache_path = make_index_cache_path(doc_path, index_root)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            return deserialize_index(pickle.load(f))
    except Exception as e:
        print(f"[WARN] Failed to load index from {cache_path}: {e}")
        return None


# ── Build / cache ─────────────────────────────────────────────────────────────

def build_doc_index(doc_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    text = read_file(doc_path)
    chunks = page_level_chunking(text)
    model = get_bge_model()
    dense_embeddings = build_dense_index(chunks=chunks, model=model) if chunks else None
    return {
        "doc_id": doc_id or os.path.basename(doc_path),
        "doc_path": doc_path,
        "chunks": chunks,
        "dense_model": model,
        "dense_embeddings": dense_embeddings,
    }


def get_or_build_doc_index(
    doc_path: str,
    index_root: str,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Load index from disk; build and save it if missing. Returns (index, was_built)."""
    idx = load_doc_index_from_disk(doc_path, index_root)
    if idx is not None:
        return idx, False

    if not os.path.exists(doc_path):
        print(f"[WARN] Doc not found when building index: {doc_path}")
        return None, False

    print(f"[INDEX] Building new index for {doc_path}")
    idx = build_doc_index(doc_path)
    save_path = save_doc_index_to_disk(idx, index_root)
    print(f"[INDEX] Saved index to {save_path}")
    return idx, True
