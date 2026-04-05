"""
/chat endpoint.

Accepts a user message and the current conversation history.
Returns the assistant's response and the updated history.
State is managed client-side; this endpoint is stateless.

If the Ollama provider is selected and the PC is offline, returns 503
with a warming-up message. The client should display this and retry.

TODO (frontend / SSE): When a frontend is built, replace the 503 with an
SSE stream. Catch ComputeWarmingUp here, switch to StreamingResponse, and
emit status events while polling compute.manager.is_pc_active(), then
stream the LLM response once online. See app/llm/ollama.py for the TODO.
"""

from llm.provider import get_provider
from fastapi import APIRouter, HTTPException, Header #type: ignore
from pydantic import BaseModel #type: ignore

from config import settings
from compute.manager import record_chat

router = APIRouter()

SYSTEM_PROMPT = """You are Trevor, a conversational AI assistant with access to Dan's
personal travel data collected over a multi-year trip across approximately nine countries.
Your data sources include GPS and location history, health metrics, financial records,
and personal journal entries.
 
Answer questions honestly and concisely, grounded in the data available to you.
If you don't know something or the data doesn't cover it, say so clearly.
"""


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

def _build_messages(request: ChatRequest) -> list[dict]:
    """
    Assemble the full message list in OpenAI format:
      [system, ...history, current user message]
 
    TODO: once tool-calling is wired up, inject tool definitions here.
    TODO: map reasoning_freedom to temperature and extend system prompt accordingly.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in request.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.message})
    return messages


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_api_key: str = Header(default=""),
):
    if x_api_key != settings.trevor_api_key:
        raise HTTPException(status_code=401, detail="Unauthorised")

    # Import here to avoid circular imports at module load time
    from llm.ollama import ComputeWarmingUp

    try:
        messages = _build_messages(request)
        provider = get_provider()
        result = await provider.chat(messages)
        response_text = result.get("content", "")
 
        record_chat()
 
        updated_history = list(request.history) + [
            Message(role="user", content=request.message),
            Message(role="assistant", content=response_text),
        ]
 
        return ChatResponse(response=response_text, history=updated_history)

    except ComputeWarmingUp as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "compute_warming_up",
                "message": str(e),
            },
        )