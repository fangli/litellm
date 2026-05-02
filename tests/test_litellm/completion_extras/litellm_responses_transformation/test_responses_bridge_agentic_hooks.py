from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponseFunctionToolCall,
    ResponsesAPIResponse,
)
from litellm.types.utils import Choices, Message, ModelResponse


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.__class__ = LiteLLMLoggingObj
    logging_obj.model_call_details = {}
    return logging_obj


def _responses_tool_call_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_test",
        created_at=1234567890,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model="gpt-5.4-mini",
        object="response",
        output=[
            ResponseFunctionToolCall(
                id="fc_1",
                type="function_call",
                status="completed",
                arguments='{"query":"NBA news"}',
                call_id="call_search",
                name="litellm_web_search",
            )
        ],
        parallel_tool_calls=True,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(
            input_tokens=50,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=10,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=60,
        ),
        user=None,
        store=True,
        background=False,
    )


def _responses_text_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_text",
        created_at=1234567890,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        model="gpt-5.4-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="Pong",
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(
            input_tokens=50,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=10,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=60,
        ),
        user=None,
        store=True,
        background=False,
    )


def _final_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="search result"),
            )
        ],
        model="gpt-5.4-mini",
        object="chat.completion",
        created=1234567890,
    )


def test_responses_bridge_runs_chat_agentic_hooks(monkeypatch):
    bridge = ResponsesToCompletionBridgeHandler()
    final_response = _final_model_response()
    hook = MagicMock(return_value=final_response)
    monkeypatch.setattr(
        ResponsesToCompletionBridgeHandler,
        "_call_agentic_chat_completion_hooks_sync",
        hook,
    )
    monkeypatch.setattr(
        litellm, "responses", MagicMock(return_value=_responses_tool_call_response())
    )

    logging_obj = _make_logging_obj()

    result = bridge.completion(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "search"}],
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ],
            "tool_choice": "auto",
        },
        litellm_params={"custom_llm_provider": "openai"},
        headers={},
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )

    assert result is final_response
    hook.assert_called_once()
    assert hook.call_args.kwargs["custom_llm_provider"] == "openai"
    assert hook.call_args.kwargs["response"].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_responses_bridge_wraps_completed_response_for_stream_request(
    monkeypatch,
):
    bridge = ResponsesToCompletionBridgeHandler()
    monkeypatch.setattr(
        litellm,
        "aresponses",
        AsyncMock(return_value=_responses_text_response()),
    )

    logging_obj = _make_logging_obj()
    result = await bridge.acompletion(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Ping"}],
        optional_params={},
        litellm_params={"custom_llm_provider": "openai"},
        headers={},
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        stream=True,
    )

    assert hasattr(result, "__aiter__")
    assert not isinstance(result, ModelResponse)
    assert logging_obj.stream is True
    assert logging_obj.model_call_details["stream"] is True
