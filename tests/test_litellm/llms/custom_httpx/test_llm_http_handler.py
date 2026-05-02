import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import litellm
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Choices, Message, ModelResponse


class _RepeatingAgenticCallback(CustomLogger):
    async def async_should_run_chat_completion_agentic_loop(self, **kwargs):
        return True, {
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "litellm_web_search",
                        "arguments": '{"query":"NBA news"}',
                    },
                }
            ]
        }

    async def async_build_chat_completion_agentic_loop_plan(self, **kwargs):
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[{"role": "user", "content": "search again"}],
                optional_params={"tools": kwargs["optional_params"].get("tools", [])},
            ),
        )

    async def async_should_run_agentic_loop(self, **kwargs):
        return True, {
            "tool_calls": [
                {
                    "id": "toolu_search",
                    "type": "tool_use",
                    "name": "litellm_web_search",
                    "input": {"query": "NBA news"},
                }
            ]
        }

    async def async_build_agentic_loop_plan(self, **kwargs):
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[{"role": "user", "content": "search again"}],
                optional_params={
                    "tools": kwargs["anthropic_messages_optional_request_params"].get(
                        "tools", []
                    )
                },
            ),
        )


def test_prepare_fake_stream_request():
    # Initialize the BaseLLMHTTPHandler
    handler = BaseLLMHTTPHandler()

    # Test case 1: fake_stream is True
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that "stream" key is removed from data
    assert "stream" not in result_data
    # Verify other data remains unchanged
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 2: fake_stream is False
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = False

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream remains True
    assert result_stream is True
    # Verify that data remains unchanged
    assert "stream" in result_data
    assert result_data["stream"] is True
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 3: data doesn't have stream key but fake_stream is True
    stream = True
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that data remains unchanged (since there was no stream key to remove)
    assert "stream" not in result_data
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]


def test_get_agentic_loop_settings_defaults_and_overrides():
    handler = BaseLLMHTTPHandler()

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(kwargs={})
    assert depth == 0
    assert max_loops == 3
    assert fingerprints == []

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(
        kwargs={
            "_agentic_loop_depth": 2,
            "max_agentic_loops": 7,
            "_agentic_loop_fingerprints": ["fp-1", "fp-2"],
        }
    )
    assert depth == 2
    assert max_loops == 7
    assert fingerprints == ["fp-1", "fp-2"]


def test_fingerprint_agentic_tools_is_deterministic():
    handler = BaseLLMHTTPHandler()
    tools_a = {"tool_calls": [{"id": "1", "input": {"q": "abc"}, "name": "web_search"}]}
    tools_b = {"tool_calls": [{"name": "web_search", "input": {"q": "abc"}, "id": "1"}]}

    assert handler._fingerprint_agentic_tools(
        tools_a
    ) == handler._fingerprint_agentic_tools(tools_b)


@pytest.mark.asyncio
async def test_chat_agentic_followup_loop_guard_propagates(monkeypatch):
    handler = BaseLLMHTTPHandler()
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(role="assistant", content=None),
            )
        ],
        model="gpt-5.4-mini",
        object="chat.completion",
        created=1234567890,
    )
    optional_params = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ]
    }

    monkeypatch.setattr(litellm, "callbacks", [_RepeatingAgenticCallback()])

    async def fake_acompletion(**kwargs):
        return await handler._call_agentic_chat_completion_hooks(
            response=response,
            model="gpt-5.4-mini",
            messages=kwargs["messages"],
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=False,
            custom_llm_provider="openai",
            kwargs=kwargs,
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(ValueError, match="Agentic loop detected repeated"):
        await handler._call_agentic_chat_completion_hooks(
            response=response,
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "search"}],
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )


@pytest.mark.asyncio
async def test_anthropic_agentic_followup_loop_guard_propagates(monkeypatch):
    handler = BaseLLMHTTPHandler()
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}
    response = {
        "id": "msg_search",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_search",
                "name": "litellm_web_search",
                "input": {"query": "NBA news"},
            }
        ],
        "stop_reason": "tool_use",
    }
    optional_params = {
        "max_tokens": 1024,
        "tools": [{"name": "litellm_web_search", "input_schema": {"type": "object"}}],
    }

    monkeypatch.setattr(litellm, "callbacks", [_RepeatingAgenticCallback()])

    async def fake_acreate(**kwargs):
        return await handler._call_agentic_completion_hooks(
            response=response,
            model="claude-haiku-4-5-20251001",
            messages=kwargs["messages"],
            anthropic_messages_provider_config=Mock(),
            anthropic_messages_optional_request_params=optional_params,
            logging_obj=logging_obj,
            stream=False,
            custom_llm_provider="bedrock",
            kwargs=kwargs,
        )

    monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", fake_acreate)

    with pytest.raises(ValueError, match="Agentic loop detected repeated"):
        await handler._call_agentic_completion_hooks(
            response=response,
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "search"}],
            anthropic_messages_provider_config=Mock(),
            anthropic_messages_optional_request_params=optional_params,
            logging_obj=logging_obj,
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_preserves_deployment_credentials_for_agentic_hooks():
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "deployment-key"}, "http://deployment.local")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude", "messages": []}
    )
    mock_config.get_complete_url = Mock(
        return_value="http://deployment.local/v1/messages"
    )
    mock_config.sign_request = Mock(return_value=({"x-api-key": "deployment-key"}, {}))
    mock_config.transform_anthropic_messages_response = Mock(
        return_value={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude",
            "stop_reason": "end_turn",
        }
    )

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    handler._async_post_anthropic_messages_with_http_error_retry = AsyncMock(
        return_value=Mock()
    )
    handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers",
        return_value=None,
    ):
        await handler.async_anthropic_messages_handler(
            model="claude",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="kiro",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            api_key="deployment-key",
            api_base="http://deployment.local",
            stream=False,
            kwargs={"temperature": 0.2},
        )

    hook_kwargs = handler._call_agentic_completion_hooks.call_args.kwargs["kwargs"]
    assert hook_kwargs["api_key"] == "deployment-key"
    assert hook_kwargs["api_base"] == "http://deployment.local"
    assert hook_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_call_agentic_completion_hooks_wraps_converted_stream_agentic_response(
    monkeypatch,
):
    handler = BaseLLMHTTPHandler()

    final_response = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "current fact from search"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    class AgenticResponseLogger(CustomLogger):
        async def async_should_run_agentic_loop(self, **kwargs):
            return True, {"tool_calls": [{"name": "litellm_web_search"}]}

        async def async_run_agentic_loop(self, **kwargs):
            return final_response

    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }

    monkeypatch.setattr(litellm, "callbacks", [AgenticResponseLogger()])

    result = await handler._call_agentic_completion_hooks(
        response={
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "litellm_web_search",
                    "input": {"query": "OpenAI news"},
                }
            ],
            "stop_reason": "tool_use",
        },
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "search"}],
        anthropic_messages_provider_config=Mock(),
        anthropic_messages_optional_request_params={
            "tools": [
                {"name": "litellm_web_search", "input_schema": {"type": "object"}}
            ]
        },
        logging_obj=logging_obj,
        stream=False,
        custom_llm_provider="chatgpt",
        kwargs={},
    )

    assert isinstance(result, FakeAnthropicMessagesStreamIterator)
    assert logging_obj.stream is True
    assert logging_obj.model_call_details["stream"] is True
    assert any(b"current fact from search" in chunk for chunk in result.chunks)


@pytest.mark.asyncio
async def test_call_agentic_completion_hooks_wraps_converted_stream_terminated_plan(
    monkeypatch,
):
    handler = BaseLLMHTTPHandler()
    initial_response = {
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [{"type": "text", "text": "partial answer"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    class TerminatingAgenticLogger(CustomLogger):
        async def async_should_run_agentic_loop(self, **kwargs):
            return True, {"tool_calls": [{"name": "litellm_web_search"}]}

        async def async_build_agentic_loop_plan(self, **kwargs):
            return AgenticLoopPlan(terminate=True, stop_reason="no_followup")

    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }

    monkeypatch.setattr(litellm, "callbacks", [TerminatingAgenticLogger()])

    result = await handler._call_agentic_completion_hooks(
        response=initial_response,
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "search"}],
        anthropic_messages_provider_config=Mock(),
        anthropic_messages_optional_request_params={
            "tools": [
                {"name": "litellm_web_search", "input_schema": {"type": "object"}}
            ]
        },
        logging_obj=logging_obj,
        stream=False,
        custom_llm_provider="chatgpt",
        kwargs={},
    )

    assert isinstance(result, FakeAnthropicMessagesStreamIterator)
    assert logging_obj.stream is True
    assert logging_obj.model_call_details["stream"] is True
    assert any(b"partial answer" in chunk for chunk in result.chunks)


@pytest.mark.asyncio
async def test_call_agentic_chat_completion_hooks_wraps_converted_stream_agentic_response(
    monkeypatch,
):
    handler = BaseLLMHTTPHandler()

    final_response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="current fact from search",
                ),
            )
        ],
        model="claude-haiku-4-5-20251001",
        object="chat.completion",
        created=1234567890,
    )

    class ChatAgenticResponseLogger(CustomLogger):
        async def async_should_run_chat_completion_agentic_loop(self, **kwargs):
            return True, {"tool_calls": [{"name": "litellm_web_search"}]}

        async def async_run_chat_completion_agentic_loop(self, **kwargs):
            return final_response

    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {
        "websearch_interception_converted_stream": True,
    }
    logging_obj.stream_options = None
    logging_obj.messages = []
    logging_obj.completion_start_time = None
    logging_obj._update_completion_start_time = Mock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = Mock()

    monkeypatch.setattr(litellm, "callbacks", [ChatAgenticResponseLogger()])

    result = await handler._call_agentic_chat_completion_hooks(
        response=ModelResponse(
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
        ),
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "search"}],
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ]
        },
        logging_obj=logging_obj,
        stream=False,
        custom_llm_provider="kiro",
        kwargs={},
    )

    assert hasattr(result, "__aiter__")
    assert logging_obj.stream is True
    assert logging_obj.model_call_details["stream"] is True
    first_chunk = await result.__anext__()
    assert first_chunk.choices[0].delta.content == "current fact from search"


@pytest.mark.asyncio
async def test_execute_chat_completion_agentic_plan_deduplicates_followup_kwargs(
    monkeypatch,
):
    handler = BaseLLMHTTPHandler()
    captured_kwargs = {}

    async def mock_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return ModelResponse()

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    await handler._execute_chat_completion_agentic_plan(
        plan=AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[
                    {"role": "user", "content": "search"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "litellm_web_search",
                                    "arguments": '{"query":"NBA news"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "NBA result",
                    },
                ],
                kwargs={"max_retries": 2, "temperature": 0.2},
            ),
        ),
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "search"}],
        optional_params={"max_retries": 2, "tools": []},
        kwargs={"max_retries": 2, "api_base": "https://api.openai.com/v1"},
        custom_llm_provider="openai",
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp-1",
    )

    assert captured_kwargs["max_retries"] == 2
    assert captured_kwargs["temperature"] == 0.2
    assert captured_kwargs["api_base"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_execute_anthropic_agentic_plan_deduplicates_followup_kwargs(
    monkeypatch,
):
    handler = BaseLLMHTTPHandler()
    captured_kwargs = {}

    async def mock_acreate(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "id": "msg_final",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
        }

    monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", mock_acreate)

    logging_obj = Mock()
    logging_obj.model_call_details = {
        "agentic_loop_params": {"model": "kiro/claude-haiku-4-5-20251001"}
    }

    await handler._execute_anthropic_agentic_plan(
        plan=AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                messages=[
                    {"role": "user", "content": "search"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "litellm_web_search",
                                "input": {"query": "NBA news"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "NBA result",
                            }
                        ],
                    },
                ],
                max_tokens=2048,
                optional_params={"tools": []},
                kwargs={"max_tokens": 4096, "temperature": 0.2},
            ),
        ),
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "search"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "tools": [{"name": "litellm_web_search"}],
        },
        logging_obj=logging_obj,
        kwargs={"max_tokens": 1024, "api_base": "http://deployment.local"},
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp-1",
        stream=False,
    )

    assert captured_kwargs["max_tokens"] == 2048
    assert captured_kwargs["temperature"] == 0.2
    assert captured_kwargs["api_base"] == "http://deployment.local"
    assert captured_kwargs["model"] == "kiro/claude-haiku-4-5-20251001"


def test_sync_completion_runs_agentic_chat_completion_hooks():
    handler = BaseLLMHTTPHandler()
    provider_config = Mock()
    provider_config.should_fake_stream.return_value = False
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://example.test/chat"
    provider_config.transform_request.return_value = {"messages": []}
    provider_config.sign_request.return_value = ({}, None)

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
    provider_config.transform_response.return_value = initial_response

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    handler._make_common_sync_call = Mock(return_value=Mock())
    handler._call_agentic_chat_completion_hooks_sync = Mock(return_value=final_response)

    result = handler.completion(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "search"}],
        api_base=None,
        custom_llm_provider="kiro",
        model_response=ModelResponse(),
        encoding=None,
        logging_obj=logging_obj,
        optional_params={"tools": []},
        timeout=10,
        litellm_params={},
        acompletion=False,
        stream=False,
        provider_config=provider_config,
    )

    assert result is final_response
    handler._call_agentic_chat_completion_hooks_sync.assert_called_once()
    assert (
        handler._call_agentic_chat_completion_hooks_sync.call_args.kwargs["response"]
        is initial_response
    )


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_extra_headers():
    """
    Test that async_anthropic_messages_handler correctly extracts and merges
    extra_headers from kwargs with proper priority.
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-3-opus-20240229", "messages": []}
    )

    # Mock the client
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    # Mock logging object
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test case 1: Only extra_headers in kwargs
    kwargs = {
        "extra_headers": {
            "X-Custom-Header": "from-kwargs",
            "X-Auth-Token": "token123",
        }
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = None

        # Capture what headers are passed to validate_anthropic_messages_environment
        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass  # We're testing header extraction, not the full flow

        # Verify extra_headers were extracted and merged
        assert "X-Custom-Header" in captured_headers
        assert captured_headers["X-Custom-Header"] == "from-kwargs"
        assert "X-Auth-Token" in captured_headers
        assert captured_headers["X-Auth-Token"] == "token123"


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_passes_litellm_metadata():
    """Ensure litellm_metadata from kwargs is forwarded via update_from_kwargs.

    Routes like /messages store model_info under kwargs['litellm_metadata'].
    The handler must forward this so that use_custom_pricing_for_model can
    detect custom pricing. Regression test for #23185.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    custom_model_info = {
        "id": "claude-sonnet-4-custom-pricing",
        "input_cost_per_token": 0.0003,
        "output_cost_per_token": 0.0015,
    }
    kwargs = {
        "litellm_metadata": {
            "model_info": custom_model_info,
            "deployment": "anthropic/claude-sonnet-4-20250514",
        },
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs=kwargs,
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    kwargs_arg = (
        call_kwargs.kwargs.get("kwargs", call_kwargs[1].get("kwargs", {}))
        if call_kwargs.kwargs
        else call_kwargs[1].get("kwargs", {})
    )

    assert "litellm_metadata" in kwargs_arg
    assert kwargs_arg["litellm_metadata"]["model_info"] == custom_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_header_priority():
    """
    Test that async_anthropic_messages_handler respects header priority:
    forwarded < extra_headers < provider_specific
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_client = AsyncMock()
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test with all three header sources
    kwargs = {
        "headers": {"X-Priority": "forwarded", "X-Forwarded-Only": "keep"},
        "extra_headers": {"X-Priority": "extra", "X-Extra-Only": "also-keep"},
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = {
            "X-Priority": "provider",
            "X-Provider-Only": "keep-this-too",
        }

        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate
        mock_config.transform_anthropic_messages_request = Mock(
            return_value={"model": "claude-3-opus-20240229", "messages": []}
        )

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass

        # Verify priority: provider_specific should win
        assert captured_headers["X-Priority"] == "provider"
        # Verify all unique headers from different sources are present
        assert captured_headers["X-Forwarded-Only"] == "keep"
        assert captured_headers["X-Extra-Only"] == "also-keep"
        assert captured_headers["X-Provider-Only"] == "keep-this-too"
