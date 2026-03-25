import asyncio
from typing import Any, Dict, List

from autogen_ext.models.openai import OpenAIChatCompletionClient

from .agents import make_finance_agents, run_agent_and_get_text
from .retrieval import build_pages_markdown, retrieve_pages_with_index


def extract_first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def is_final(eval_text: str) -> bool:
    return (eval_text or "").strip().upper() == "NONE"


async def run_multi_round_rag_with_index(
    question: str,
    index: Dict[str, Any],
    model_client: OpenAIChatCompletionClient,
    semaphore: asyncio.Semaphore,
    max_rounds: int = 4,
    top_k: int = 6,
) -> Dict[str, Any]:
    """
    Multi-round RAG loop (FinLongDocAgent):

    Each round:
      1. ExpansionAgent generates a formula / account query.
      2. Dense retrieval fetches the top-k pages.
      3. SolvingAgent computes the answer from retrieved pages.
      4. EvaluationAgent checks for missing components or wrong accounts.
      5. If eval returns 'NONE', stop; otherwise feed missing hints to the next round.
    """
    expansion_agent, solving_agent, evaluation_agent = make_finance_agents(model_client)

    expanded_formulas: List[str] = []
    eval_traces: List[str] = []
    missing_components_note: str = ""
    last_pages: List[Dict[str, Any]] = []
    sol_msg: str = ""

    for r in range(1, max_rounds + 1):
        # 1) Expansion
        expansion_input = (
            question
            if not missing_components_note
            else f"{question}\n\nMissing components from last round: {missing_components_note}"
        )
        exp_raw = await run_agent_and_get_text(expansion_agent, expansion_input, semaphore)
        formula = extract_first_nonempty_line(exp_raw)
        expanded_formulas.append(formula or "<EMPTY>")

        # 2) Retrieval
        pages = retrieve_pages_with_index(formula, index=index, top_k=top_k)
        last_pages = pages

        # 3) Solving
        solving_context = (
            f"# Question\n{question}\n\n"
            f"# Expanded Formula (from ExpansionAgent)\n{formula}\n\n"
            f"# Retrieved Pages (markdown)\n{build_pages_markdown(pages)}\n\n"
            "Use the pages above only. Show your calculation clearly with page_number and formula.\n"
            "Return the final numeric answer rounded to two decimals in double curly braces {{like this}} with no extra text.\n"
            "If insufficient information, return {{0}}.\n\n"
            "Given a user's question about a financial ratio in an annual report, think of which line items (accounts) will feed into that formula, and *only* return the complete formula as a whole-no explanations."
        )
        sol_msg = await run_agent_and_get_text(solving_agent, solving_context, semaphore)

        # 4) Evaluation
        eval_context = (
            f"Question:\n{question}\n\n"
            f"Proposed Answer (from SolvingAgent):\n{sol_msg}\n"
        )
        ev_msg = await run_agent_and_get_text(evaluation_agent, eval_context, semaphore)
        eval_text = (ev_msg or "").strip()
        eval_traces.append(eval_text)

        if is_final(eval_text):
            break

        missing_components_note = eval_text

    return {
        "answer": sol_msg,
        "rounds": r,
        "expanded_formulas": expanded_formulas,
        "eval_traces": eval_traces,
        "last_pages_used": last_pages,
    }
