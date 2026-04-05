"""
LLM provider abstraction.

All LLM calls go through get_provider(), which returns the correct
backend based on LLM_PROVIDER env var. Swap providers without touching
any other part of the codebase.

Decision log: Provider is configurable from day one to avoid lock-in.
Ollama (local) is used during development; OpenAI/Anthropic for production
or when model quality matters for a demo.
"""

from config import settings
from llm.ollama import OllamaProvider
from llm.openai import OpenAIProvider
from llm.anthropic import AnthropicProvider


def get_provider():
    """Return the configured LLM provider instance."""
    match settings.llm_provider:
        case "ollama":
            return OllamaProvider()
        case "openai":
            return OpenAIProvider()
        case "anthropic":
            return AnthropicProvider()
        case _:
            raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
