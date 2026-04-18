"""
OpenAI provider.

Uses openai.AsyncOpenAI for both chat and embeddings.
Supports native tool/function calling — tool definitions are passed
directly to the API and tool_calls are returned in the response dict
for the tool-calling loop in chat.py to handle.

Model defaults (set via config):
  Chat:   gpt-5.4-nano  (cheapest current model; swap to gpt-5.4 for demos)
  Embed:  text-embedding-3-small

Prompt caching: OpenAI automatically caches repeated prompt prefixes
(system prompt + schema) at $0.02/1M tokens. No extra work needed.
"""

import json
import openai as openai_lib
from llm.base import BaseLLMProvider
from config import settings


class OpenAIProvider(BaseLLMProvider):

    def _client(self) -> openai_lib.AsyncOpenAI:
        return openai_lib.AsyncOpenAI(api_key=settings.openai_api_key)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        kwargs = {
            "model": settings.openai_model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client().chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Normalise tool calls into a list of dicts for the tool-calling loop
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
            "assistant_message": message.model_dump(),
        }

    async def embed(self, text: str) -> list[float]:
        response = await self._client().embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding