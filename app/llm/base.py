"""
Abstract base class for LLM providers.
All providers must implement chat() and embed().
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send a chat request.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional list of tool definitions (function-calling).
            temperature: Sampling temperature.

        Returns:
            Dict with at minimum:
                "content": str — the assistant's text response
                "tool_calls": list — any tool calls requested (may be empty)
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Embed a string and return the vector.
        Used for journal ingestion and query-time retrieval.
        """
        ...
