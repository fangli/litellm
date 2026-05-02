from unittest.mock import AsyncMock, Mock

import litellm
import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.experimental_pass_through.responses_adapters import handler
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    LiteLLMMessagesToResponsesAPIHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse


@pytest.mark.asyncio
async def test_responses_adapter_runs_anthropic_agentic_hook(monkeypatch):
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
    monkeypatch.setattr(
        handler.litellm,
        "aresponses",
        AsyncMock(return_value=ResponsesAPIResponse.model_construct(output=[])),
    )
    monkeypatch.setattr(
        handler._ADAPTER,
        "translate_response",
        Mock(return_value=tool_response),
    )

    result = (
        await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=1024,
            messages=[{"role": "user", "content": "search"}],
            model="openai/gpt-5.4-mini",
            tools=[{"name": "litellm_web_search", "input_schema": {"type": "object"}}],
            custom_llm_provider="openai",
        )
    )

    assert result == final_response
