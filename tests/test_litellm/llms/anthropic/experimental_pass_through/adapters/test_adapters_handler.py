from unittest.mock import AsyncMock, Mock

import litellm
import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.experimental_pass_through.adapters import handler
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


@pytest.mark.asyncio
async def test_completion_adapter_runs_anthropic_agentic_hook(monkeypatch):
    tool_response = {
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "litellm_web_search",
                "input": {"query": "OpenAI news"},
            }
        ],
        "stop_reason": "tool_use",
    }
    final_response = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "search answer"}],
        "stop_reason": "end_turn",
    }

    class AgenticResponseLogger(CustomLogger):
        async def async_should_run_agentic_loop(self, response, **kwargs):
            assert response == tool_response
            return True, {"tool_calls": [{"name": "litellm_web_search"}]}

        async def async_run_agentic_loop(self, **kwargs):
            return final_response

    monkeypatch.setattr(litellm, "callbacks", [AgenticResponseLogger()])
    monkeypatch.setattr(handler.litellm, "acompletion", AsyncMock(return_value=Mock()))
    monkeypatch.setattr(
        handler.ANTHROPIC_ADAPTER,
        "translate_completion_output_params",
        Mock(return_value=tool_response),
    )

    result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=[{"role": "user", "content": "search"}],
        model="chatgpt/gpt-5.4-mini",
        tools=[{"name": "litellm_web_search", "input_schema": {"type": "object"}}],
        custom_llm_provider="chatgpt",
    )

    assert result == final_response


@pytest.mark.asyncio
async def test_completion_adapter_branches_on_prepared_stream_flag(monkeypatch):
    stream_response = Mock()
    monkeypatch.setattr(
        LiteLLMMessagesToCompletionTransformationHandler,
        "_prepare_completion_kwargs",
        Mock(return_value=({"model": "chatgpt/gpt-5.4-mini", "stream": True}, {})),
    )
    monkeypatch.setattr(
        handler.litellm,
        "acompletion",
        AsyncMock(return_value=stream_response),
    )
    streaming_transform = Mock(return_value="anthropic-stream")
    non_streaming_transform = Mock()
    monkeypatch.setattr(
        handler.ANTHROPIC_ADAPTER,
        "translate_completion_output_params_streaming",
        streaming_transform,
    )
    monkeypatch.setattr(
        handler.ANTHROPIC_ADAPTER,
        "translate_completion_output_params",
        non_streaming_transform,
    )

    result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Ping"}],
        model="chatgpt/gpt-5.4-mini",
        custom_llm_provider="chatgpt",
    )

    assert result == "anthropic-stream"
    streaming_transform.assert_called_once_with(
        stream_response,
        model="chatgpt/gpt-5.4-mini",
        tool_name_mapping={},
    )
    non_streaming_transform.assert_not_called()


@pytest.mark.asyncio
async def test_completion_adapter_branches_on_streaming_response_shape(monkeypatch):
    class AsyncStream:
        def __aiter__(self):
            return self

    stream_response = AsyncStream()
    monkeypatch.setattr(
        LiteLLMMessagesToCompletionTransformationHandler,
        "_prepare_completion_kwargs",
        Mock(return_value=({"model": "chatgpt/gpt-5.4-mini"}, {})),
    )
    monkeypatch.setattr(
        handler.litellm,
        "acompletion",
        AsyncMock(return_value=stream_response),
    )
    streaming_transform = Mock(return_value="anthropic-stream")
    non_streaming_transform = Mock()
    monkeypatch.setattr(
        handler.ANTHROPIC_ADAPTER,
        "translate_completion_output_params_streaming",
        streaming_transform,
    )
    monkeypatch.setattr(
        handler.ANTHROPIC_ADAPTER,
        "translate_completion_output_params",
        non_streaming_transform,
    )

    result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Ping"}],
        model="chatgpt/gpt-5.4-mini",
        custom_llm_provider="chatgpt",
    )

    assert result == "anthropic-stream"
    streaming_transform.assert_called_once()
    non_streaming_transform.assert_not_called()
