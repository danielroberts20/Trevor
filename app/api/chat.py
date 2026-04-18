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

import logging

from llm.ollama import ComputeWarmingUp
from llm.provider import get_provider
from fastapi import APIRouter, HTTPException, Header, Request #type: ignore
from pydantic import BaseModel #type: ignore
from datetime import date
import json
from config import settings
from compute.manager import record_chat
from tools.search_journal import TOOL_DEFINITION as SEARCH_JOURNAL_TOOL
from tools.query_db import TOOL_DEFINITION as QUERY_DB_TOOL
from prompt import SYSTEM_PROMPT_BASE, SCHEMA_BLOCK_TEMPLATE

logger = logging.getLogger(__name__)

router = APIRouter()


TOOLS = [SEARCH_JOURNAL_TOOL, QUERY_DB_TOOL]


def _build_system_prompt(db_schema: str) -> str:
    """
    Build the system prompt, injecting the DB schema if available.
    If the schema failed to load at startup, the schema block is omitted
    and the LLM will indicate it cannot query structured data.
    """
    today = date.today().isoformat()
    date_line = f"Today's date: {today}.\n"

    if db_schema:
        schema_block = SCHEMA_BLOCK_TEMPLATE.format(schema=db_schema)
    else:
        schema_block = "(Database schema unavailable — structured queries are disabled.)"

    return SYSTEM_PROMPT_BASE.format(date_line=date_line, schema_block=schema_block)


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

def _build_messages(request: ChatRequest, db_schema: str) -> list[dict]:
    """
    Assemble the full message list in OpenAI format:
      [system, ...history, current user message]
 
    TODO: once tool-calling is wired up, inject tool definitions here.
    TODO: map reasoning_freedom to temperature and extend system prompt accordingly.
    """
    system_prompt = _build_system_prompt(db_schema)
    messages = [{"role": "system", "content": system_prompt}]
    for m in request.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.message})
    return messages

MAX_TOOL_ITERATIONS = 5

async def _run_turn(messages: list[dict], provider, temperature: float) -> str:
    for i in range(MAX_TOOL_ITERATIONS):
        is_last = i == MAX_TOOL_ITERATIONS - 1

        if is_last:
            messages.append({
                "role": "user",
                "content": (
                    "Please stop calling tools and give your best answer "
                    "using the information you have already retrieved."
                ),
            })

        result = await provider.chat(messages, tools=TOOLS, temperature=temperature)
        finish_reason = result.get("finish_reason")

        if finish_reason == "stop":
            return result["content"]

        if finish_reason == "tool_calls":
            messages.append(result["assistant_message"])
            for tool_call in result["tool_calls"]:
                tool_result = _dispatch_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result),
                })
            continue

        logger.warning("Unexpected finish_reason: %s", finish_reason)
        return result.get("content", "")

    logger.warning("Tool-calling loop exhausted MAX_TOOL_ITERATIONS without stop")
    return "I wasn't able to complete that request."


def _dispatch_tool(tool_call: dict) -> dict:
    name = tool_call["name"]
    args = tool_call["arguments"]  # already parsed from JSON by provider

    if name == "query_db":
        from retrieval.db_client import query
        return query(**args)
    if name == "search_journal":
        from tools.search_journal import search
        return search(**args)

    return {"error": f"Unknown tool: {name}"}

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: str = Header(default=""),
):
    if x_api_key != settings.trevor_api_key:
        raise HTTPException(status_code=401, detail="Unauthorised")
 
    try:
        db_schema = getattr(raw_request.app.state, "db_schema", "")
        messages = _build_messages(request, db_schema)
        provider = get_provider()
        response_text = await _run_turn(messages, provider, temperature=request.reasoning_freedom / 10)
        logger.info("LLM response: %s", response_text)
 
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