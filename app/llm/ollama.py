"""
Ollama provider — calls the custom FastAPI wrapper on the PC (port 8100).

Endpoints:
  POST /chat  {"message": str, "model": str} -> {"response": str}
  POST /embed {"text": str}                  -> {"embedding": [...]}
"""

import httpx
from llm.base import BaseLLMProvider
from config import settings

TIMEOUT = 60.0  # Ollama can be slow on first load; be generous


class OllamaProvider(BaseLLMProvider):

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        # TODO: Ollama's /chat wrapper currently accepts a single "message" string.
        # For multi-turn history, either:
        #   a) concatenate history into a single prompt string here, or
        #   b) update the PC-side wrapper to accept OpenAI-format messages.
        # Option (b) is cleaner long-term.
        last_user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{settings.ollama_base_url}/chat",
                json={"message": last_user_message, "model": settings.ollama_model},
            )
            r.raise_for_status()
            return {"content": r.json()["response"], "tool_calls": []}

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{settings.ollama_base_url}/embed",
                json={"text": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]
