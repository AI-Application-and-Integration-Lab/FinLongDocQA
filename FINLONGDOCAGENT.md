# FinLongDocAgent

**Multi-Agent Multi-Round RAG for Financial Long-Document QA**

FinLongDocAgent is the baseline method introduced alongside the FinLongDocQA benchmark.
It addresses two bottlenecks in long financial document QA:

1. **Context rot** — annual reports exceed 129k tokens, making relevant table location unreliable with full-context approaches.
2. **Multi-step numerical errors** — even with the right evidence, LLMs make arithmetic mistakes in cross-table reasoning.

FinLongDocAgent tackles both through iterative retrieval and verification across multiple rounds.

## Architecture

```
For each question:
  Round 1..N:
    ExpansionAgent  →  generates a formula/account query
         ↓
    Dense Retrieval  →  BAAI/bge-m3 retrieves top-k pages
         ↓
    SolvingAgent    →  computes the answer from retrieved pages
         ↓
    EvaluationAgent →  checks for missing components or wrong accounts
         ↓
    if NONE → stop   else → feed missing hints into Round N+1
```

| Agent | Role |
|---|---|
| **ExpansionAgent** | Translates the question into financial formula terms for retrieval |
| **SolvingAgent** | Answers using retrieved pages; shows page references and formula |
| **EvaluationAgent** | Identifies missing evidence or account mismatches; returns `NONE` when satisfied |

## Installation

```bash
pip install autogen-agentchat autogen-ext sentence-transformers numpy tqdm
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Document Format

Annual reports must be stored as markdown files under `reports/`:

```
reports/
  AAPL/
    2022.md
    2023.md
  MSFT/
    2022.md
```

Each file should use `# Page N` headings so the system can split and index by page.

## Running

```bash
python run.py \
    --query-path  dataset_qa.jsonl \
    --doc-root    reports \
    --output-path results/output.jsonl \
    --index-root  dense_indexes \
    --max-rounds  5 \
    --top-k       15 \
    --max-concurrent-llm 4
```

| Argument | Default | Description |
|---|---|---|
| `--query-path` | `dataset_qa.jsonl` | Input JSONL with questions |
| `--doc-root` | `reports` | Root directory of `<company>/<year>.md` files |
| `--output-path` | `results/output.jsonl` | Output JSONL for predictions |
| `--index-root` | `dense_indexes` | Cache directory for per-document dense indexes |
| `--max-rounds` | `5` | Maximum refinement rounds per question |
| `--top-k` | `15` | Pages retrieved per round |
| `--max-concurrent-llm` | `4` | Concurrent LLM calls (semaphore size) |

Runs are **resumable**: if `--output-path` already exists, completed questions are skipped automatically.

## Output Format

Each line of the output JSONL contains:

```json
{
  "company": "AAPL",
  "year": "2023",
  "question": "What was the Return on Assets for fiscal year 2023?",
  "answer": "The ROA is {{11.65}}",
  "rounds": 2,
  "expanded_formulas": ["net income / average total assets", "..."],
  "eval_traces": ["net income, total assets", "NONE"],
  "last_pages_used": [
    {"doc_id": "2023.md", "page_number": 57, "content": "...", "score": 0.91}
  ]
}
```

## Package Structure

```
finlongdocagent/
├── config.py          # Lazy OpenRouter model client (get_model_client())
├── prompts.py         # System prompts for the three agents
├── embeddings.py      # Lazy BAAI/bge-m3 model (get_bge_model()), dense index/retrieve
├── io_utils.py        # JSONL and file I/O helpers
├── indexing.py        # Page chunking, index build, disk cache (pickle)
├── retrieval.py       # Dense retrieval, page markdown builder
├── agents.py          # AutoGen agent factory and runner
├── pipeline.py        # Multi-round RAG loop
└── orchestration.py   # Resume logic, batch indexing, async main()
```

## Using as a Library

```python
import asyncio
from finlongdocagent.indexing import get_or_build_doc_index
from finlongdocagent.pipeline import run_multi_round_rag_with_index
from finlongdocagent.config import get_model_client

async def run_one(question: str, doc_path: str):
    index, _ = get_or_build_doc_index(doc_path, index_root="dense_indexes")
    result = await run_multi_round_rag_with_index(
        question=question,
        index=index,
        model_client=get_model_client(),
        semaphore=asyncio.Semaphore(1),
        max_rounds=5,
        top_k=15,
    )
    print(result["answer"])

asyncio.run(run_one("What was the ROA in 2023?", "reports/AAPL/2023.md"))
```
