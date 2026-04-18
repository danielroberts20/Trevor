"""
test_chat.py — Unit and integration tests for api/chat.py.

Covers:
  _build_system_prompt():
    - Includes today's date in ISO format
    - Includes the schema block when db_schema is non-empty
    - Includes the unavailable fallback when db_schema is empty
    - Does not raise on either input

  _build_messages():
    - First message has role "system"
    - History messages appear in order between system and the new user message
    - Last message has role "user" and content equal to request.message
    - History roles are preserved exactly

  POST /chat endpoint:
    - Returns 401 when x-api-key header is missing or wrong
    - Does not call the LLM provider on auth failure
    - Returns 200 with "response" (str) and "history" (list) on success
    - Updated history includes the new user message and the assistant reply
    - Returns 503 with error="compute_warming_up" when ComputeWarmingUp is raised
    - Stateless: two requests with different histories return independent results

  _run_turn():
    - Returns a string (the assistant's final response)
    - Calls provider.chat() at least once
    - Returns immediately on finish_reason="stop" without a second provider call
    - Dispatches the tool and calls provider.chat() again on finish_reason="tool_calls"
    - Appends the assistant message (with tool_calls) before appending the tool result
    - Does not call provider.chat() more than MAX_TOOL_ITERATIONS times
    - Returns a non-empty fallback string when MAX_TOOL_ITERATIONS is reached
    - Handles multiple tool calls in a single response

  _dispatch_tool():
    - Routes "query_db" to retrieval.db_client.query
    - Routes "search_journal" to tools.search_journal.search
    - Returns {"error": "Unknown tool: <name>"} for unrecognised names
    - Passes arguments as keyword arguments (already a dict — no JSON parsing)
"""

import json
import pytest
from datetime import date
from unittest.mock import patch, MagicMock, AsyncMock

from api.chat import (
    _build_system_prompt,
    _build_messages,
    _run_turn,
    _dispatch_tool,
    ChatRequest,
    Message,
    MAX_TOOL_ITERATIONS,
)
from llm.ollama import ComputeWarmingUp
from tests.conftest import TEST_API_KEY


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:

    def test_build_system_prompt_includes_todays_date(self):
        result = _build_system_prompt("some schema")
        today = date.today().isoformat()
        assert today in result

    def test_build_system_prompt_includes_schema_block_when_schema_provided(self):
        result = _build_system_prompt("CREATE TABLE trips(id INTEGER);")
        assert "CREATE TABLE trips(id INTEGER);" in result

    def test_build_system_prompt_includes_unavailable_message_when_schema_empty(self):
        result = _build_system_prompt("")
        assert "unavailable" in result.lower()

    def test_build_system_prompt_does_not_raise_with_nonempty_schema(self):
        _build_system_prompt("CREATE TABLE foo(id INTEGER);")  # must not raise

    def test_build_system_prompt_does_not_raise_with_empty_schema(self):
        _build_system_prompt("")  # must not raise

    def test_build_system_prompt_schema_block_absent_when_schema_empty(self):
        # When the schema is empty the LLM should not receive a garbled schema block
        result = _build_system_prompt("")
        assert "CREATE TABLE" not in result


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:

    def _make_request(self, message, history=None):
        history = history or []
        return ChatRequest(message=message, history=history)

    def test_build_messages_first_message_is_system(self):
        request = self._make_request("Hello")
        messages = _build_messages(request, "some schema")
        assert messages[0]["role"] == "system"

    def test_build_messages_last_message_is_user(self):
        request = self._make_request("Hello")
        messages = _build_messages(request, "some schema")
        assert messages[-1]["role"] == "user"

    def test_build_messages_last_message_content_equals_request_message(self):
        request = self._make_request("What's my step count?")
        messages = _build_messages(request, "some schema")
        assert messages[-1]["content"] == "What's my step count?"

    def test_build_messages_history_appears_between_system_and_user(self):
        history = [
            Message(role="user",      content="First question"),
            Message(role="assistant", content="First answer"),
        ]
        request = self._make_request("Second question", history=history)
        messages = _build_messages(request, "some schema")

        # system at 0, history at 1..n-1, new user message at -1
        assert messages[1]["content"] == "First question"
        assert messages[2]["content"] == "First answer"
        assert messages[-1]["content"] == "Second question"

    def test_build_messages_history_order_preserved(self):
        history = [
            Message(role="user",      content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user",      content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        request = self._make_request("Q3", history=history)
        messages = _build_messages(request, "some schema")

        contents = [m["content"] for m in messages[1:-1]]
        assert contents == ["Q1", "A1", "Q2", "A2"]

    def test_build_messages_history_roles_preserved_exactly(self):
        history = [
            Message(role="user",      content="Q"),
            Message(role="assistant", content="A"),
        ]
        request = self._make_request("follow-up", history=history)
        messages = _build_messages(request, "some schema")

        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_build_messages_with_empty_history(self):
        request = self._make_request("Solo question")
        messages = _build_messages(request, "some schema")
        # system + user only
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# POST /chat endpoint
# ---------------------------------------------------------------------------

class TestChatEndpoint:

    def _auth_header(self):
        return {"x-api-key": TEST_API_KEY}

    def test_chat_endpoint_missing_api_key_returns_401(self, client):
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_chat_endpoint_wrong_api_key_returns_401(self, client):
        resp = client.post("/chat", json={"message": "hello"}, headers={"x-api-key": "wrong"})
        assert resp.status_code == 401

    def test_chat_endpoint_auth_failure_does_not_call_provider(self, client):
        with patch("api.chat.get_provider") as mock_get_provider:
            client.post("/chat", json={"message": "hello"}, headers={"x-api-key": "wrong"})
        mock_get_provider.assert_not_called()

    def test_chat_endpoint_success_returns_200(self, client, mock_provider):
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Hello"},
                headers=self._auth_header(),
            )
        assert resp.status_code == 200

    def test_chat_endpoint_success_response_has_response_key(self, client, mock_provider):
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Hello"},
                headers=self._auth_header(),
            )
        assert "response" in resp.json()
        assert isinstance(resp.json()["response"], str)

    def test_chat_endpoint_success_response_has_history_key(self, client, mock_provider):
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Hello"},
                headers=self._auth_header(),
            )
        assert "history" in resp.json()
        assert isinstance(resp.json()["history"], list)

    def test_chat_endpoint_history_includes_new_user_message(self, client, mock_provider):
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "What is my step count?"},
                headers=self._auth_header(),
            )
        history = resp.json()["history"]
        user_messages = [m for m in history if m["role"] == "user"]
        assert any(m["content"] == "What is my step count?" for m in user_messages)

    def test_chat_endpoint_history_includes_assistant_response(self, client, mock_provider):
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Hello"},
                headers=self._auth_header(),
            )
        history = resp.json()["history"]
        assistant_messages = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1

    def test_chat_endpoint_history_appended_to_input_history(self, client, mock_provider):
        input_history = [
            {"role": "user",      "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Follow-up", "history": input_history},
                headers=self._auth_header(),
            )
        history = resp.json()["history"]
        # The two input messages should still be in the returned history
        assert history[0]["content"] == "Earlier question"
        assert history[1]["content"] == "Earlier answer"

    def test_chat_endpoint_503_on_compute_warming_up(self, client):
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(side_effect=ComputeWarmingUp("PC is offline"))

        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp = client.post(
                "/chat",
                json={"message": "Hello"},
                headers=self._auth_header(),
            )
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "compute_warming_up"

    def test_chat_endpoint_is_stateless(self, client, mock_provider):
        # Two requests with different histories must get independent responses;
        # the server must not bleed state from one request into the next.
        with patch("api.chat.get_provider", return_value=mock_provider), \
             patch("api.chat.record_chat"):
            resp1 = client.post(
                "/chat",
                json={"message": "Q1", "history": []},
                headers=self._auth_header(),
            )
            resp2 = client.post(
                "/chat",
                json={
                    "message": "Q2",
                    "history": [{"role": "user", "content": "Prior"}],
                },
                headers=self._auth_header(),
            )

        h1 = resp1.json()["history"]
        h2 = resp2.json()["history"]

        # First request: only Q1 + assistant reply
        assert len(h1) == 2
        assert h1[0]["content"] == "Q1"

        # Second request: Prior + Q2 + assistant reply
        assert len(h2) == 3
        assert h2[0]["content"] == "Prior"
        assert h2[1]["content"] == "Q2"


# ---------------------------------------------------------------------------
# _run_turn
# ---------------------------------------------------------------------------

class TestRunTurn:

    def _stop_response(self, content="Assistant reply"):
        return {
            "content": content,
            "finish_reason": "stop",
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": content},
        }

    def _tool_response(self, tool_calls):
        return {
            "content": "",
            "finish_reason": "tool_calls",
            "tool_calls": tool_calls,
            "assistant_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        }

    @pytest.mark.asyncio
    async def test_run_turn_returns_string(self, mock_provider):
        messages = [{"role": "user", "content": "Hello"}]
        result = await _run_turn(messages, mock_provider, temperature=0.5)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_turn_calls_provider_chat(self, mock_provider):
        messages = [{"role": "user", "content": "Hello"}]
        await _run_turn(messages, mock_provider, temperature=0.5)
        mock_provider.chat.assert_called()

    @pytest.mark.asyncio
    async def test_run_turn_stop_finish_reason_returns_content(self, mock_provider):
        mock_provider.chat.return_value = self._stop_response("The answer is 42.")
        messages = [{"role": "user", "content": "Hello"}]
        result = await _run_turn(messages, mock_provider, temperature=0.5)
        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_run_turn_stop_does_not_call_provider_second_time(self, mock_provider):
        mock_provider.chat.return_value = self._stop_response()
        messages = [{"role": "user", "content": "Hello"}]
        await _run_turn(messages, mock_provider, temperature=0.5)
        assert mock_provider.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_run_turn_dispatches_tool_on_tool_calls(self, mock_provider):
        tool_call = {"id": "tc1", "name": "query_db", "arguments": {"sql": "SELECT 1"}}
        mock_provider.chat = AsyncMock(side_effect=[
            self._tool_response([tool_call]),
            self._stop_response("Done"),
        ])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}) as mock_dispatch:
            await _run_turn(messages, mock_provider, temperature=0.5)

        mock_dispatch.assert_called_once_with(tool_call)

    @pytest.mark.asyncio
    async def test_run_turn_appends_assistant_message_before_tool_result(self, mock_provider):
        tool_call = {"id": "tc1", "name": "query_db", "arguments": {"sql": "SELECT 1"}}
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        mock_provider.chat = AsyncMock(side_effect=[
            {
                "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [tool_call],
                "assistant_message": assistant_msg,
            },
            self._stop_response("Done"),
        ])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}):
            await _run_turn(messages, mock_provider, temperature=0.5)

        # After the loop: messages[-2] must be assistant (with tool_calls),
        # messages[-1] must be the tool result — order matters for OpenAI.
        assert messages[-2] == assistant_msg
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "tc1"

    @pytest.mark.asyncio
    async def test_run_turn_calls_provider_again_after_tool_call(self, mock_provider):
        tool_call = {"id": "tc1", "name": "query_db", "arguments": {"sql": "SELECT 1"}}
        mock_provider.chat = AsyncMock(side_effect=[
            self._tool_response([tool_call]),
            self._stop_response("Done"),
        ])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}):
            await _run_turn(messages, mock_provider, temperature=0.5)

        assert mock_provider.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_run_turn_stops_after_max_iterations(self, mock_provider):
        tool_call = {"id": "tc1", "name": "query_db", "arguments": {"sql": "SELECT 1"}}
        mock_provider.chat = AsyncMock(
            return_value=self._tool_response([tool_call])
        )
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}):
            await _run_turn(messages, mock_provider, temperature=0.5)

        assert mock_provider.chat.call_count == MAX_TOOL_ITERATIONS

    @pytest.mark.asyncio
    async def test_run_turn_returns_nonempty_fallback_when_max_iterations_reached(
        self, mock_provider
    ):
        tool_call = {"id": "tc1", "name": "query_db", "arguments": {"sql": "SELECT 1"}}
        mock_provider.chat = AsyncMock(
            return_value=self._tool_response([tool_call])
        )
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}):
            result = await _run_turn(messages, mock_provider, temperature=0.5)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_run_turn_handles_multiple_tool_calls_in_single_response(self, mock_provider):
        tool_calls = [
            {"id": "tc1", "name": "query_db",     "arguments": {"sql": "SELECT 1"}},
            {"id": "tc2", "name": "query_db",     "arguments": {"sql": "SELECT 2"}},
        ]
        mock_provider.chat = AsyncMock(side_effect=[
            self._tool_response(tool_calls),
            self._stop_response("Done"),
        ])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("api.chat._dispatch_tool", return_value={"rows": []}) as mock_dispatch:
            await _run_turn(messages, mock_provider, temperature=0.5)

        # Both tool calls must be dispatched
        assert mock_dispatch.call_count == 2
        # Both tool results must appear in messages
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2


# ---------------------------------------------------------------------------
# _dispatch_tool
# ---------------------------------------------------------------------------

class TestDispatchTool:

    def test_dispatch_tool_routes_query_db(self):
        tool_call = {
            "id": "tc1",
            "name": "query_db",
            "arguments": {"sql": "SELECT 1"},
        }
        with patch("retrieval.db_client.query", return_value={"columns": [], "rows": []}) as mock_q:
            _dispatch_tool(tool_call)
        mock_q.assert_called_once_with(sql="SELECT 1")

    def test_dispatch_tool_routes_search_journal(self):
        tool_call = {
            "id": "tc2",
            "name": "search_journal",
            "arguments": {"query": "Melbourne", "n_results": 3},
        }
        # tools.search_journal.search does not yet exist; create=True adds it for
        # the duration of this test so the routing can be verified.
        with patch("tools.search_journal.search", create=True, return_value=[]) as mock_s:
            _dispatch_tool(tool_call)
        mock_s.assert_called_once_with(query="Melbourne", n_results=3)

    def test_dispatch_tool_returns_error_for_unknown_tool(self):
        tool_call = {
            "id": "tc3",
            "name": "explode_database",
            "arguments": {},
        }
        result = _dispatch_tool(tool_call)
        assert "error" in result
        assert "explode_database" in result["error"]

    def test_dispatch_tool_passes_arguments_as_kwargs(self):
        # arguments must already be a dict; _dispatch_tool must NOT attempt to
        # parse JSON — that is the provider's responsibility.
        tool_call = {
            "id": "tc4",
            "name": "query_db",
            "arguments": {"sql": "SELECT 2", "row_limit": 50},
        }
        with patch("retrieval.db_client.query", return_value={}) as mock_q:
            _dispatch_tool(tool_call)
        mock_q.assert_called_once_with(sql="SELECT 2", row_limit=50)
