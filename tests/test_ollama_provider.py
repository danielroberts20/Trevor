"""
test_ollama_provider.py — Unit tests for llm/ollama.py OllamaProvider.chat().

Covers:
  - Returns dict with keys: content, finish_reason, tool_calls
  - finish_reason is always "stop"
  - tool_calls is always an empty list
  - Raises ComputeWarmingUp (does not return) when the PC is offline
  - Does not call wake_pc() if the PC is already online
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from llm.ollama import OllamaProvider, ComputeWarmingUp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_client_mock(response_text="Hello from Ollama"):
    """Build a mock httpx.AsyncClient that returns a fake Ollama response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": response_text}
    mock_response.raise_for_status = MagicMock()  # does not raise

    mock_http_client = AsyncMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)
    return mock_http_client


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestOllamaChatReturnStructure:

    @pytest.mark.asyncio
    async def test_chat_returns_expected_keys(self):
        mock_http_client = _make_http_client_mock()
        with patch("llm.ollama.is_pc_active", return_value=True), \
             patch("llm.ollama.wake_pc"), \
             patch("httpx.AsyncClient") as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__  = AsyncMock(return_value=None)

            provider = OllamaProvider()
            result = await provider.chat([{"role": "user", "content": "Hello"}])

        assert "content"      in result
        assert "finish_reason" in result
        assert "tool_calls"   in result

    @pytest.mark.asyncio
    async def test_chat_content_is_string(self):
        mock_http_client = _make_http_client_mock("The answer is 42.")
        with patch("llm.ollama.is_pc_active", return_value=True), \
             patch("llm.ollama.wake_pc"), \
             patch("httpx.AsyncClient") as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__  = AsyncMock(return_value=None)

            provider = OllamaProvider()
            result = await provider.chat([{"role": "user", "content": "Hello"}])

        assert isinstance(result["content"], str)
        assert result["content"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_chat_finish_reason_is_always_stop(self):
        mock_http_client = _make_http_client_mock()
        with patch("llm.ollama.is_pc_active", return_value=True), \
             patch("llm.ollama.wake_pc"), \
             patch("httpx.AsyncClient") as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__  = AsyncMock(return_value=None)

            provider = OllamaProvider()
            result = await provider.chat([{"role": "user", "content": "Hello"}])

        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_tool_calls_is_always_empty_list(self):
        mock_http_client = _make_http_client_mock()
        with patch("llm.ollama.is_pc_active", return_value=True), \
             patch("llm.ollama.wake_pc"), \
             patch("httpx.AsyncClient") as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__  = AsyncMock(return_value=None)

            provider = OllamaProvider()
            result = await provider.chat([{"role": "user", "content": "Hello"}])

        assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# PC state handling
# ---------------------------------------------------------------------------

class TestOllamaChatPcState:

    @pytest.mark.asyncio
    async def test_chat_raises_compute_warming_up_when_pc_offline(self):
        # When the PC is offline, _ensure_pc_online() must raise ComputeWarmingUp
        # before any HTTP request is attempted.
        with patch("llm.ollama.is_pc_active", return_value=False), \
             patch("llm.ollama.wake_pc"):
            provider = OllamaProvider()
            with pytest.raises(ComputeWarmingUp):
                await provider.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_chat_does_not_call_wake_pc_when_pc_online(self):
        mock_http_client = _make_http_client_mock()
        with patch("llm.ollama.is_pc_active", return_value=True), \
             patch("llm.ollama.wake_pc") as mock_wake, \
             patch("httpx.AsyncClient") as mock_class:
            mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_class.return_value.__aexit__  = AsyncMock(return_value=None)

            provider = OllamaProvider()
            await provider.chat([{"role": "user", "content": "Hello"}])

        mock_wake.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_calls_wake_pc_when_pc_offline(self):
        # When the PC is offline, wake_pc() must be called before raising
        with patch("llm.ollama.is_pc_active", return_value=False), \
             patch("llm.ollama.wake_pc") as mock_wake:
            provider = OllamaProvider()
            with pytest.raises(ComputeWarmingUp):
                await provider.chat([{"role": "user", "content": "Hello"}])

        mock_wake.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_does_not_return_when_pc_offline(self):
        # Must raise, never return a dict, when the PC is offline
        with patch("llm.ollama.is_pc_active", return_value=False), \
             patch("llm.ollama.wake_pc"):
            provider = OllamaProvider()
            raised = False
            try:
                await provider.chat([{"role": "user", "content": "Hello"}])
            except ComputeWarmingUp:
                raised = True
        assert raised, "ComputeWarmingUp must be raised when PC is offline"
