import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_model_client: OpenAIChatCompletionClient | None = None


def get_model_client() -> OpenAIChatCompletionClient:
    global _model_client
    if _model_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Please set OPENROUTER_API_KEY in your environment.")
        _model_client = OpenAIChatCompletionClient(
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
            api_key=api_key,
            temperature=0.0,
            timeout=360,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "structured_output": False,
                "multiple_system_messages": True,
                "family": ModelFamily.UNKNOWN,
            },
        )
    return _model_client
