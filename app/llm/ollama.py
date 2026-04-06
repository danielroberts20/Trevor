"""
Ollama provider — calls the custom FastAPI wrapper on the PC (port 8100).

Before every call, checks whether the PC is online via compute.manager.
If the PC is asleep, sends a wake request and raises ComputeWarmingUp —
the /chat endpoint catches this and returns a 503 to the client.

The PC-side wrapper currently accepts a single "message" string, so the
full message list (system + history + user) is concatenated into one
prompt. This is sufficient for end-to-end testing.

TODO (frontend / SSE): Replace the ComputeWarmingUp 503 response with an
SSE stream. See app/api/chat.py.

Endpoints (PC-side FastAPI wrapper):
  POST /chat  {"message": str, "model": str} -> {"response": str}
  POST /embed {"text": str}                  -> {"embedding": [...]}
"""

import httpx
from llm.base import BaseLLMProvider
from compute.manager import is_pc_active, wake_pc
from config import settings

TIMEOUT = 60.0  # Ollama can be slow on first load; be generous


class ComputeWarmingUp(Exception):
    """Raised when the PC is offline and a wake has been triggered."""
    pass


class OllamaProvider(BaseLLMProvider):

    def _ensure_pc_online(self) -> None:
        """Check PC state; wake and raise if offline."""
        if not is_pc_active():
            wake_pc()
            raise ComputeWarmingUp(
                "Compute service is offline — a wake request has been sent. "
                "Please try again in a moment."
            )

    async def chat(self, 
                   messages: list[dict], 
                   tools=None, 
                   temperature=0.7) -> dict:
        self._ensure_pc_online()
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{settings.ollama_base_url}/chat",
                json={"messages": messages, "model": settings.ollama_model},
            )
            r.raise_for_status()
            return {"content": r.json()["response"], "tool_calls": []}

    async def embed(self, text: str) -> list[float]:
        self._ensure_pc_online()

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{settings.ollama_base_url}/embed",
                json={"text": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]