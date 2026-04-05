"""
/chat endpoint.

Accepts a user message and the current conversation history.
Returns the assistant's response and the updated history.
State is managed client-side; this endpoint is stateless.
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from config import settings

router = APIRouter()


class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    reasoning_freedom: int = 5  # 0–10 slider; see Reasoning Freedom spec


class ChatResponse(BaseModel):
    response: str
    history: list[Message]


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_api_key: str = Header(default=""),
):
    # Basic API key check — dual-layer security model mirrors query_db() SELECT enforcement
    if x_api_key != settings.trevor_api_key:
        raise HTTPException(status_code=401, detail="Unauthorised")

    # TODO: wire up tool-calling pipeline
    # 1. build tool list: [search_journal, query_db]
    # 2. call llm/provider.py chat() with history + tools
    # 3. handle tool_use blocks -> dispatch to tools/ -> re-enter LLM with results
    # 4. return final text response

    placeholder_response = (
        f"[Trevor stub] Received: '{request.message}' | "
        f"Provider: {settings.llm_provider} | "
        f"Reasoning freedom: {request.reasoning_freedom}/10"
    )

    updated_history = list(request.history) + [
        Message(role="user", content=request.message),
        Message(role="assistant", content=placeholder_response),
    ]

    return ChatResponse(response=placeholder_response, history=updated_history)
