from unittest.mock import MagicMock

from litellm.llms.anthropic.chat import handler as anthropic_chat_handler
from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from litellm.types.utils import Choices, Message, ModelResponse
from litellm.utils import ProviderConfigManager


def test_anthropic_chat_marks_websearch_converted_stream():
    logging_obj = MagicMock()
    logging_obj.stream = True
    logging_obj.model_call_details = {}

    AnthropicChatCompletion._mark_websearch_converted_stream_if_needed(
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


def test_anthropic_chat_completion_runs_agentic_hooks_after_transform(monkeypatch):
    completion = AnthropicChatCompletion()
    initial_response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(role="assistant", content=None),
            )
        ],
        model="claude-haiku-4-5-20251001",
        object="chat.completion",
        created=1234567890,
    )
    final_response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="search result"),
            )
        ],
        model="claude-haiku-4-5-20251001",
        object="chat.completion",
        created=1234567890,
    )

    config = MagicMock()
    config.transform_request.return_value = {"messages": []}
    config.transform_response.return_value = initial_response
    monkeypatch.setattr(
        ProviderConfigManager,
        "get_provider_chat_config",
        MagicMock(return_value=config),
    )
    monkeypatch.setattr(
        anthropic_chat_handler.AnthropicConfig,
        "validate_environment",
        MagicMock(return_value={}),
    )

    http_client = MagicMock()
    http_client.post.return_value = MagicMock()
    monkeypatch.setattr(
        anthropic_chat_handler,
        "_get_httpx_client",
        MagicMock(return_value=http_client),
    )
    hook = MagicMock(return_value=final_response)
    monkeypatch.setattr(
        AnthropicChatCompletion,
        "_run_agentic_chat_completion_hooks_sync",
        hook,
    )

    logging_obj = MagicMock()
    logging_obj.stream = False
    logging_obj.model_call_details = {}

    result = completion.completion(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "search"}],
        api_base="https://api.anthropic.com/v1/messages",
        custom_llm_provider="anthropic",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=lambda *_args, **_kwargs: None,
        encoding=None,
        api_key="test-key",
        logging_obj=logging_obj,
        optional_params={"tools": []},
        timeout=10,
        litellm_params={},
        acompletion=False,
    )

    assert result is final_response
    hook.assert_called_once()
    assert hook.call_args.kwargs["response"] is initial_response
