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

from config import settings
from compute.manager import record_chat

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_PROMPT_BASE = """You are Trevor, a conversational AI assistant with access to Dan's \
personal travel data collected over a multi-year trip across approximately nine countries. \
Your data sources include GPS and location history, health metrics, financial records, \
and personal journal entries.
 
Answer questions honestly and concisely, grounded in the data available to you. \
If you don't know something or the data doesn't cover it, say so clearly.

You have access to two tools:

search_journal(query: str, n_results: int)
  Search Dan's travel journal entries semantically. Use this for questions about
  experiences, feelings, events, reflections, or anything narrative in nature.
  Examples: "how did I feel in Vietnam", "what happened in my first week in Japan"

query_db(sql: str)
  Execute a SELECT query against the TravelNet database. Use this for precise
  structured questions about location history, health metrics, spending, and
  ML model outputs (HMM segments, DBSCAN clusters, anomaly flags).
  Only SELECT statements are permitted.
  Examples: "what was my average daily spend in Thailand", "how many steps on 14 March"
 
{schema_block}

Rules:
- Always use a tool if the question requires data. Never guess or fabricate data.
- Use search_journal for narrative questions, query_db for structured ones.
- For cross-stream questions (e.g. "did my mood correlate with spending?"), use both.
- Cite sources using <CITE:chunk_id> for journal entries.
- Tag speculative content with <SPEC>...</SPEC>."""
 
SCHEMA_BLOCK_TEMPLATE = """You have access to the TravelNet database. \
Only SELECT queries are permitted. The schema is as follows:
 
{schema}"""

from tools.search_journal import TOOL_DEFINITION as SEARCH_JOURNAL_TOOL
from tools.query_db import TOOL_DEFINITION as QUERY_DB_TOOL

TOOLS = [SEARCH_JOURNAL_TOOL, QUERY_DB_TOOL]
 
 
def _build_system_prompt(db_schema: str) -> str:
    """
    Build the system prompt, injecting the DB schema if available.
    If the schema failed to load at startup, the schema block is omitted
    and the LLM will indicate it cannot query structured data.
    """
    if db_schema:
        schema_block = SCHEMA_BLOCK_TEMPLATE.format(schema=db_schema)
    else:
        schema_block = "(Database schema unavailable — structured queries are disabled.)"
 
    return SYSTEM_PROMPT_BASE.format(schema_block=schema_block)


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
        result = await provider.chat(messages, tools=TOOLS, temperature=request.reasoning_freedom / 10)
        logger.info("LLM raw response: %s", result)
        response_text = result["content"]
 
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