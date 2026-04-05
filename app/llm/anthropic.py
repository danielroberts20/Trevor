"""
Anthropic provider stub.
Uses the anthropic SDK; tool use maps to Anthropic's tool_use content blocks.
"""

from llm.base import BaseLLMProvider
from config import settings


class AnthropicProvider(BaseLLMProvider):

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        # TODO: implement using anthropic.AsyncAnthropic
        # Note: Anthropic uses "tools" with input_schema rather than OpenAI's "parameters"
        # Normalise tool definitions in provider.py or here before passing to SDK
        raise NotImplementedError

    async def embed(self, text: str) -> list[float]:
        # Anthropic does not offer a first-party embedding endpoint.
        # Options: use OpenAI embeddings regardless of chat provider,
        # or use nomic-embed-text via Ollama.
        # Decision to document when embedding strategy is finalised.
        raise NotImplementedError
