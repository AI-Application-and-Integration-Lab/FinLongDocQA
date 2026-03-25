from typing import Any, Dict, List

from .embeddings import dense_retrieve_with_index


def retrieve_pages_with_index(
    query: str,
    index: Dict[str, Any],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    chunks = index["chunks"]
    doc_id = index["doc_id"]
    if not chunks:
        return []

    hits = dense_retrieve_with_index(query, index=index, top_k=top_k)
    return [
        {
            "doc_id": doc_id,
            "page_number": i + 1,
            "content": chunks[i],
            "score": score,
        }
        for i, score in hits
        if 0 <= i < len(chunks)
    ]


def build_pages_markdown(pages: List[Dict[str, Any]]) -> str:
    if not pages:
        return "_No relevant pages found._"
    parts = []
    for p in pages:
        header = f"### {p.get('doc_id', 'doc')} – Page {p['page_number']}"
        parts.append(f"{header}\n{p['content']}")
    return "\n\n".join(parts)
