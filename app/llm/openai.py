"""
OpenAI provider stub.
Uses the openai SDK; supports tool/function calling natively.
"""

from llm.base import BaseLLMProvider
from config import settings


class OpenAIProvider(BaseLLMProvider):

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        # TODO: implement using openai.AsyncOpenAI
        # client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        # response = await client.chat.completions.create(...)
        raise NotImplementedError

    async def embed(self, text: str) -> list[float]:
        # TODO: implement using text-embedding-3-small
        raise NotImplementedError
