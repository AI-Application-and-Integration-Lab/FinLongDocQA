"""
Entry point for FinLongDocAgent.

Usage:
    python run.py \
        --query-path  dataset_qa.jsonl \
        --doc-root    reports \
        --output-path results/output.jsonl \
        --index-root  dense_indexes \
        --max-rounds  5 \
        --top-k       15 \
        --max-concurrent-llm 4
"""

import argparse
import asyncio

from finlongdocagent.orchestration import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinLongDocAgent: Multi-Agent Multi-Round RAG for financial long-document QA."
    )
    parser.add_argument(
        "--query-path",
        type=str,
        default="dataset_qa.jsonl",
        help="Path to input JSONL queries file.",
    )
    parser.add_argument(
        "--doc-root",
        type=str,
        default="reports",
        help="Root directory containing <company>/<year>.md annual reports.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/output.jsonl",
        help="Output JSONL path for predictions.",
    )
    parser.add_argument(
        "--index-root",
        type=str,
        default="dense_indexes",
        help="Directory where per-document dense indexes are cached.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of RAG refinement rounds per question.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of pages to retrieve per round.",
    )
    parser.add_argument(
        "--max-concurrent-llm",
        type=int,
        default=4,
        help="Maximum concurrent LLM calls (semaphore size).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
