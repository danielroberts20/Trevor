"""
test_openai_provider.py — Unit tests for llm/openai.py OpenAIProvider.chat().

Covers:
  - Returns dict with keys: content, finish_reason, tool_calls, assistant_message
  - tool_calls is a list of dicts with keys: id, name, arguments
  - arguments is a dict, not a JSON string (provider normalises it)
  - assistant_message is the model_dump() of the raw message object
  - content is an empty string (not None) when the response contains tool_calls
    and no text content
  - tool_choice is "auto" when tools are provided to the API call
  - tool_choice is absent from kwargs when no tools are provided
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stop_response(content="Hello, world!"):
    """Mock openai response object for a simple stop completion."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call_response(tool_calls_data: list[dict]):
    """
    Mock openai response object with tool_calls.
    tool_calls_data: list of {"id": ..., "name": ..., "arguments_json": ...}
    """
    mock_tool_calls = []
    for tc in tool_calls_data:
        mock_tc = MagicMock()
        mock_tc.id = tc["id"]
        mock_tc.function.name = tc["name"]
        mock_tc.function.arguments = tc["arguments_json"]
        mock_tool_calls.append(mock_tc)

    message = MagicMock()
    message.content = None
    message.tool_calls = mock_tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": tc["id"]} for tc in tool_calls_data],
    }

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_provider_with_response(mock_response):
    """Return (provider, mock_create) with chat.completions.create mocked."""
    mock_create = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    return mock_client, mock_create


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestOpenAIChatReturnStructure:

    @pytest.mark.asyncio
    async def test_chat_returns_expected_keys(self):
        mock_client, _ = _make_provider_with_response(_make_stop_response())
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])
        assert "content"           in result
        assert "finish_reason"     in result
        assert "tool_calls"        in result
        assert "assistant_message" in result

    @pytest.mark.asyncio
    async def test_chat_content_is_string(self):
        mock_client, _ = _make_provider_with_response(_make_stop_response("Reply"))
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result["content"], str)
        assert result["content"] == "Reply"

    @pytest.mark.asyncio
    async def test_chat_finish_reason_is_string(self):
        mock_client, _ = _make_provider_with_response(_make_stop_response())
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result["finish_reason"], str)

    @pytest.mark.asyncio
    async def test_chat_tool_calls_is_list(self):
        mock_client, _ = _make_provider_with_response(_make_stop_response())
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result["tool_calls"], list)

    @pytest.mark.asyncio
    async def test_chat_assistant_message_is_dict(self):
        mock_client, _ = _make_provider_with_response(_make_stop_response())
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result["assistant_message"], dict)


# ---------------------------------------------------------------------------
# Tool calls normalisation
# ---------------------------------------------------------------------------

class TestOpenAIChatToolCalls:

    @pytest.mark.asyncio
    async def test_chat_tool_calls_each_have_id_name_arguments(self):
        mock_response = _make_tool_call_response([
            {"id": "call_abc", "name": "query_db", "arguments_json": '{"sql": "SELECT 1"}'},
        ])
        mock_client, _ = _make_provider_with_response(mock_response)
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Query something"}])

        tc = result["tool_calls"][0]
        assert "id"        in tc
        assert "name"      in tc
        assert "arguments" in tc

    @pytest.mark.asyncio
    async def test_chat_tool_call_arguments_is_dict_not_string(self):
        # The provider must parse the JSON string from the API into a dict.
        mock_response = _make_tool_call_response([
            {"id": "call_xyz", "name": "query_db", "arguments_json": '{"sql": "SELECT 2"}'},
        ])
        mock_client, _ = _make_provider_with_response(mock_response)
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Query something"}])

        arguments = result["tool_calls"][0]["arguments"]
        assert isinstance(arguments, dict), "arguments must be a dict, not a JSON string"
        assert arguments["sql"] == "SELECT 2"

    @pytest.mark.asyncio
    async def test_chat_tool_call_id_and_name_preserved(self):
        mock_response = _make_tool_call_response([
            {"id": "call_123", "name": "search_journal", "arguments_json": '{"query": "Australia"}'},
        ])
        mock_client, _ = _make_provider_with_response(mock_response)
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Search journals"}])

        tc = result["tool_calls"][0]
        assert tc["id"]   == "call_123"
        assert tc["name"] == "search_journal"

    @pytest.mark.asyncio
    async def test_chat_content_is_empty_string_not_none_when_tool_calls_present(self):
        # When the API returns None content (tool-only response), the provider
        # must normalise it to "" — callers must not receive None.
        mock_response = _make_tool_call_response([
            {"id": "tc1", "name": "query_db", "arguments_json": '{"sql": "SELECT 1"}'},
        ])
        mock_client, _ = _make_provider_with_response(mock_response)
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Q"}])

        assert result["content"] == ""
        assert result["content"] is not None


# ---------------------------------------------------------------------------
# assistant_message is model_dump() of the full message object
# ---------------------------------------------------------------------------

class TestOpenAIChatAssistantMessage:

    @pytest.mark.asyncio
    async def test_chat_assistant_message_comes_from_model_dump(self):
        expected_dump = {"role": "assistant", "content": "Hi there", "extra": "field"}
        message = MagicMock()
        message.content = "Hi there"
        message.tool_calls = None
        message.model_dump.return_value = expected_dump

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"

        response = MagicMock()
        response.choices = [choice]

        mock_client, _ = _make_provider_with_response(response)
        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            result = await provider.chat([{"role": "user", "content": "Hi"}])

        assert result["assistant_message"] == expected_dump


# ---------------------------------------------------------------------------
# tool_choice parameter
# ---------------------------------------------------------------------------

class TestOpenAIChatToolChoice:

    @pytest.mark.asyncio
    async def test_chat_tool_choice_auto_when_tools_provided(self):
        mock_client, mock_create = _make_provider_with_response(_make_stop_response())
        tools = [{"type": "function", "function": {"name": "query_db"}}]

        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            await provider.chat([{"role": "user", "content": "Hi"}], tools=tools)

        _, kwargs = mock_create.call_args
        assert kwargs.get("tool_choice") == "auto"

    @pytest.mark.asyncio
    async def test_chat_tool_choice_absent_when_no_tools(self):
        mock_client, mock_create = _make_provider_with_response(_make_stop_response())

        with patch("llm.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIProvider()
            await provider.chat([{"role": "user", "content": "Hi"}])

        _, kwargs = mock_create.call_args
        assert "tool_choice" not in kwargs
