"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_headers(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="session-123")
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] == "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert request["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ]
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    def test_chatgpt_preserves_caller_instructions(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4-mini",
            input="hi",
            response_api_optional_request_params={
                "instructions": "Use the caller supplied instructions.",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["instructions"] == "Use the caller supplied instructions."

    def test_chatgpt_preserves_empty_caller_instructions(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4-mini",
            input="hi",
            response_api_optional_request_params={"instructions": ""},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["instructions"] == ""

    def test_chatgpt_rewrites_direct_responses_system_input_to_developer(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.4-mini",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "follow policy"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"][0]["role"] == "developer"
        assert request["input"][1]["role"] == "user"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 200000}
                ],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "service_tier": "priority",
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["service_tier"] == "priority"
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    def test_chatgpt_non_stream_sse_response_reconstructs_function_call_output(self):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4-mini",
            "output": [],
        }
        sse_body = "\n".join(
            [
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": "fc_123",
                            "type": "function_call",
                            "call_id": "call_123",
                            "name": "litellm_web_search",
                            "arguments": "",
                        },
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "delta": '{"query":"LiteLLM',
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 0,
                        "delta": ' Responses"}',
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": 0,
                        "arguments": '{"query":"LiteLLM Responses"}',
                    }
                ),
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4-mini",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert len(parsed.output) == 1
        function_call = parsed.output[0]
        function_call_dict = (
            function_call.model_dump()
            if hasattr(function_call, "model_dump")
            else function_call
        )
        assert function_call_dict["type"] == "function_call"
        assert function_call_dict["name"] == "litellm_web_search"
        assert function_call_dict["call_id"] == "call_123"
        assert function_call_dict["arguments"] == '{"query":"LiteLLM Responses"}'
