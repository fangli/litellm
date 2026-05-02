from types import SimpleNamespace
from unittest.mock import MagicMock

from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.utils import Choices, Message, ModelResponse
from litellm.utils import CustomStreamWrapper


def _model_response(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content=content),
            )
        ],
        model="gpt-5.4-mini",
        object="chat.completion",
        created=1234567890,
    )


def test_openai_sync_completion_runs_agentic_hooks(monkeypatch):
    completion = OpenAIChatCompletion()
    final_response = _model_response("search result")
    hook = MagicMock(return_value=final_response)
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_call_agentic_completion_hooks_openai_sync",
        hook,
    )

    fake_client = SimpleNamespace(
        api_key="test-key",
        _base_url=SimpleNamespace(_uri_reference="https://api.openai.com/v1"),
    )
    monkeypatch.setattr(
        completion,
        "_get_openai_client",
        MagicMock(return_value=fake_client),
    )

    raw_response = MagicMock()
    raw_response.model_dump.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5.4-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "litellm_web_search",
                                "arguments": '{"query":"NBA news"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    monkeypatch.setattr(
        completion,
        "make_sync_openai_chat_completion_request",
        MagicMock(return_value=({"x-request-id": "req-test"}, raw_response)),
    )

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.stream = False

    result = completion.completion(
        model_response=ModelResponse(),
        timeout=10.0,
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ]
        },
        litellm_params={"custom_llm_provider": "openai"},
        logging_obj=logging_obj,
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "search"}],
        api_key="test-key",
        custom_llm_provider="openai",
    )

    assert result is final_response
    hook.assert_called_once()
    assert hook.call_args.kwargs["stream"] is False


def test_openai_wraps_websearch_converted_stream_response():
    completion = OpenAIChatCompletion()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }

    result = completion._maybe_wrap_websearch_converted_chat_stream_response(
        response=_model_response("streamed final text"),
        logging_obj=logging_obj,
        model="gpt-5.4-mini",
        stream_options=None,
    )

    assert isinstance(result, CustomStreamWrapper)
    assert logging_obj.stream is True
    assert logging_obj.model_call_details["stream"] is True


def test_openai_marks_websearch_converted_stream_from_request_tools():
    logging_obj = MagicMock()
    logging_obj.stream = True
    logging_obj.model_call_details = {}

    OpenAIChatCompletion._mark_websearch_converted_stream_if_needed(
        logging_obj=logging_obj,
        litellm_params={},
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ]
        },
        stream=False,
    )

    assert (
        logging_obj.model_call_details["websearch_interception_converted_stream"]
        is True
    )
