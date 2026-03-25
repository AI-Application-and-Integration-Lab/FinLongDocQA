import asyncio
import os
from typing import Any, Dict, Set, Tuple

from tqdm import tqdm

from .config import get_model_client
from .indexing import get_or_build_doc_index
from .io_utils import append_jsonl, read_jsonl
from .pipeline import run_multi_round_rag_with_index


# ── Resume helpers ────────────────────────────────────────────────────────────

def make_key(company: str, year: str, question: str) -> Tuple[str, str, str]:
    return (company, str(year), question)


def load_completed_keys(output_path: str) -> Set[Tuple[str, str, str]]:
    completed: Set[Tuple[str, str, str]] = set()
    if not os.path.exists(output_path):
        return completed
    try:
        for r in read_jsonl(output_path):
            if "company" in r and "year" in r and "question" in r:
                completed.add(make_key(r["company"], r["year"], r["question"]))
    except Exception as e:
        print(f"[WARN] Failed to read existing output for resume: {e}")
    return completed


def build_all_doc_indexes(
    query_datas: list,
    doc_root: str,
    index_root: str,
) -> Dict[str, Dict[str, Any]]:
    """Build or load dense indexes for every document referenced in the queries."""
    doc_paths = sorted({
        os.path.join(doc_root, q["company"], f'{q["year"]}.md')
        for q in query_datas
    })
    print(f"[INDEX] Preparing indexes for {len(doc_paths)} unique documents...")
    cache: Dict[str, Dict[str, Any]] = {}
    for doc_path in tqdm(doc_paths, desc="Indexing docs"):
        idx, _ = get_or_build_doc_index(doc_path, index_root=index_root)
        if idx is not None:
            cache[doc_path] = idx
    print(f"[INDEX] Ready. {len(cache)} indexes available.")
    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args) -> None:
    model_client = get_model_client()
    semaphore = asyncio.Semaphore(args.max_concurrent_llm)

    query_datas = read_jsonl(args.query_path)

    completed_keys = load_completed_keys(args.output_path)
    if completed_keys:
        print(f"[RESUME] Found {len(completed_keys)} completed items in {args.output_path}. Skipping them...")

    doc_index_cache = build_all_doc_indexes(
        query_datas=query_datas,
        doc_root=args.doc_root,
        index_root=args.index_root,
    )

    for query_data in tqdm(query_datas, desc="Answering questions"):
        company = query_data["company"]
        year = query_data["year"]
        question = query_data["question"]

        key = make_key(company, year, question)
        if key in completed_keys:
            continue

        doc_path = os.path.join(args.doc_root, company, f"{year}.md")

        if doc_path not in doc_index_cache:
            idx, _ = get_or_build_doc_index(doc_path, index_root=args.index_root)
            if idx is None:
                print(f"[WARN] No index available for doc: {doc_path} (skip)")
                continue
            doc_index_cache[doc_path] = idx

        result = await run_multi_round_rag_with_index(
            question=question,
            index=doc_index_cache[doc_path],
            model_client=model_client,
            semaphore=semaphore,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
        )

        record: Dict[str, Any] = {
            "company": company,
            "year": year,
            "question": question,
            "answer": result["answer"],
            "rounds": result["rounds"],
            "expanded_formulas": result["expanded_formulas"],
            "eval_traces": result["eval_traces"],
            "last_pages_used": result["last_pages_used"],
        }
        append_jsonl(args.output_path, record)
        completed_keys.add(key)

    await model_client.close()
    print(f"\nSaved results to: {args.output_path}")
