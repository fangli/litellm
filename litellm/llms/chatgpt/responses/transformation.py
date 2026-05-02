import json
from typing import Any, Dict, List, Optional

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    @staticmethod
    def _normalize_input_for_chatgpt(input_param: Any) -> Any:
        if isinstance(input_param, str):
            return [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_param}],
                }
            ]

        if not isinstance(input_param, list):
            return input_param

        normalized_input = []
        for item in input_param:
            if isinstance(item, dict):
                item = dict(item)
                if item.get("role") == "system":
                    item["role"] = "developer"
            normalized_input.append(item)
        return normalized_input

    @staticmethod
    def _get_event_output_index(event: Dict[str, Any]) -> int:
        output_index = event.get("output_index", 0)
        if isinstance(output_index, int):
            return output_index
        try:
            return int(output_index)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _get_event_content_index(event: Dict[str, Any]) -> int:
        content_index = event.get("content_index", 0)
        if isinstance(content_index, int):
            return content_index
        try:
            return int(content_index)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_empty_response_value(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    @classmethod
    def _merge_content_part_dict(
        cls, target_part: Dict[str, Any], source_part: Dict[str, Any]
    ) -> None:
        for key, value in source_part.items():
            if cls._is_empty_response_value(target_part.get(key)):
                target_part[key] = value

    @classmethod
    def _merge_output_item_dict(
        cls, target_item: Dict[str, Any], source_item: Dict[str, Any]
    ) -> None:
        for key, value in source_item.items():
            if key == "content" and isinstance(value, list):
                target_content = target_item.setdefault("content", [])
                if not isinstance(target_content, list):
                    continue
                for index, content_part in enumerate(value):
                    if index >= len(target_content):
                        target_content.append(content_part)
                    elif isinstance(content_part, dict) and isinstance(
                        target_content[index], dict
                    ):
                        cls._merge_content_part_dict(
                            target_content[index], content_part
                        )
                    elif cls._is_empty_response_value(target_content[index]):
                        target_content[index] = content_part
            elif key not in target_item or cls._is_empty_response_value(
                target_item.get(key)
            ):
                target_item[key] = value

    @classmethod
    def _ensure_sse_output_item(
        cls,
        accumulated_output: Dict[int, Dict[str, Any]],
        output_index: int,
        item_type: str,
    ) -> Dict[str, Any]:
        output_item = accumulated_output.setdefault(output_index, {"type": item_type})
        output_item.setdefault("type", item_type)
        if output_item.get("type") == "message":
            output_item.setdefault("role", "assistant")
            output_item.setdefault("content", [])
        if output_item.get("type") == "function_call":
            output_item.setdefault("arguments", "")
        return output_item

    @classmethod
    def _ensure_sse_content_part(
        cls, output_item: Dict[str, Any], content_index: int
    ) -> Dict[str, Any]:
        output_item.setdefault("type", "message")
        output_item.setdefault("role", "assistant")
        content = output_item.setdefault("content", [])
        while len(content) <= content_index:
            content.append({"type": "output_text", "text": ""})
        content_part = content[content_index]
        if not isinstance(content_part, dict):
            content_part = {"type": "output_text", "text": str(content_part)}
            content[content_index] = content_part
        content_part.setdefault("type", "output_text")
        content_part.setdefault("text", "")
        return content_part

    @classmethod
    def _record_sse_output_event(
        cls,
        event: Dict[str, Any],
        accumulated_output: Dict[int, Dict[str, Any]],
    ) -> None:
        event_type = event.get("type")
        output_index = cls._get_event_output_index(event)

        if event_type in (
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ):
            item = event.get("item")
            if isinstance(item, dict):
                output_item = accumulated_output.setdefault(output_index, {})
                cls._merge_output_item_dict(output_item, item)
                if item.get("type") == "message":
                    output_item.setdefault("role", "assistant")
                    output_item.setdefault("content", [])
                if item.get("type") == "function_call":
                    output_item.setdefault("arguments", "")
                    if item.get("arguments"):
                        output_item["arguments"] = item["arguments"]
            return

        if event_type in (
            ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ):
            output_item = cls._ensure_sse_output_item(
                accumulated_output, output_index, "message"
            )
            content_part = cls._ensure_sse_content_part(
                output_item, cls._get_event_content_index(event)
            )
            part = event.get("part")
            if isinstance(part, dict):
                cls._merge_content_part_dict(content_part, part)
            return

        if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            output_item = cls._ensure_sse_output_item(
                accumulated_output, output_index, "message"
            )
            content_part = cls._ensure_sse_content_part(
                output_item, cls._get_event_content_index(event)
            )
            content_part["text"] = (content_part.get("text") or "") + event.get(
                "delta", ""
            )
            return

        if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
            output_item = cls._ensure_sse_output_item(
                accumulated_output, output_index, "message"
            )
            content_part = cls._ensure_sse_content_part(
                output_item, cls._get_event_content_index(event)
            )
            if isinstance(event.get("text"), str):
                content_part["text"] = event["text"]
            return

        if event_type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            output_item = cls._ensure_sse_output_item(
                accumulated_output, output_index, "function_call"
            )
            output_item["arguments"] = (output_item.get("arguments") or "") + event.get(
                "delta", ""
            )
            return

        if event_type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE:
            output_item = cls._ensure_sse_output_item(
                accumulated_output, output_index, "function_call"
            )
            if isinstance(event.get("arguments"), str):
                output_item["arguments"] = event["arguments"]

    @classmethod
    def _merge_sse_output_into_completed_response(
        cls,
        response_payload: Dict[str, Any],
        accumulated_output: Dict[int, Dict[str, Any]],
    ) -> None:
        if not accumulated_output:
            return

        accumulated_output_items: List[Dict[str, Any]] = [
            item for _, item in sorted(accumulated_output.items()) if item
        ]
        if not accumulated_output_items:
            return

        output = response_payload.get("output")
        if not isinstance(output, list) or not output:
            response_payload["output"] = accumulated_output_items
            return

        for output_index, accumulated_item in sorted(accumulated_output.items()):
            if output_index >= len(output):
                output.append(accumulated_item)
                continue
            if isinstance(output[output_index], dict):
                cls._merge_output_item_dict(output[output_index], accumulated_item)
            elif cls._is_empty_response_value(output[output_index]):
                output[output_index] = accumulated_item

    def _parse_sse_completed_response_payload(
        self,
        raw_response: Any,
        body_text: str,
    ) -> Dict[str, Any]:
        error_message = None
        accumulated_output: Dict[int, Dict[str, Any]] = {}

        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue
            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk:
                continue
            if stripped_chunk == STREAM_SSE_DONE_STRING:
                break
            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_chunk, dict):
                continue
            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    response_payload = dict(response_payload)
                    self._merge_sse_output_into_completed_response(
                        response_payload, accumulated_output
                    )
                    return response_payload
                break
            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_obj = parsed_chunk.get("error") or (
                    parsed_chunk.get("response") or {}
                ).get("error")
                if error_obj is not None:
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get("message") or str(error_obj)
                    else:
                        error_message = str(error_obj)
            else:
                self._record_sse_output_event(parsed_chunk, accumulated_output)

        raise OpenAIError(
            message=error_message or raw_response.text,
            status_code=raw_response.status_code,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        return {**default_headers, **headers}

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        request["input"] = self._normalize_input_for_chatgpt(request.get("input"))
        base_instructions = get_chatgpt_default_instructions()
        existing_instructions = request.get("instructions")
        if existing_instructions is None:
            request["instructions"] = base_instructions
        request["store"] = False
        request["stream"] = True
        include = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "tools",
            "tool_choice",
            "reasoning",
            "previous_response_id",
            "truncation",
        }

        return {k: v for k, v in request.items() if k in allowed_keys}

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        content_type = (raw_response.headers or {}).get("content-type", "")
        body_text = raw_response.text or ""
        if "text/event-stream" not in content_type.lower():
            trimmed_body = body_text.lstrip()
            if not (
                trimmed_body.startswith("event:")
                or trimmed_body.startswith("data:")
                or "\nevent:" in body_text
                or "\ndata:" in body_text
            ):
                return super().transform_response_api_response(
                    model=model,
                    raw_response=raw_response,
                    logging_obj=logging_obj,
                )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        response_payload = self._parse_sse_completed_response_payload(
            raw_response=raw_response,
            body_text=body_text,
        )
        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(
                response_payload["created_at"]
            )
        try:
            completed_response = ResponsesAPIResponse(**response_payload)
        except Exception:
            completed_response = ResponsesAPIResponse.model_construct(
                **response_payload
            )

        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers
        return completed_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False
