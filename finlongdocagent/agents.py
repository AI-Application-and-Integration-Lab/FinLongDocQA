import asyncio
from typing import Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .prompts import EVAL_SYS_PROMPT, EXPANSION_SYS_PROMPT, SOLVING_SYS_PROMPT


def make_finance_agents(
    model_client: OpenAIChatCompletionClient,
) -> Tuple[AssistantAgent, AssistantAgent, AssistantAgent]:
    """Create fresh expansion, solving, and evaluation agents for a single question."""
    expansion_agent = AssistantAgent(
        name="ExpansionAgent",
        system_message=EXPANSION_SYS_PROMPT,
        model_client=model_client,
    )
    solving_agent = AssistantAgent(
        name="SolvingAgent",
        system_message=SOLVING_SYS_PROMPT,
        model_client=model_client,
    )
    evaluation_agent = AssistantAgent(
        name="EvaluationAgent",
        system_message=EVAL_SYS_PROMPT,
        model_client=model_client,
    )
    return expansion_agent, solving_agent, evaluation_agent


async def run_agent_and_get_text(
    agent: AssistantAgent,
    task: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Run an agent with a semaphore guard and return the last message as plain text."""
    async with semaphore:
        result = await agent.run(task=task)
    if not result.messages:
        return ""
    return str(getattr(result.messages[-1], "content", ""))
