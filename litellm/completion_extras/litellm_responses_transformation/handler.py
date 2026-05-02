"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

from typing import TYPE_CHECKING, Any, Coroutine, Optional, Union

from typing_extensions import TypedDict

from litellm.types.llms.openai import ResponsesAPIResponse

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper, LiteLLMLoggingObj, ModelResponse


class ResponsesToCompletionBridgeHandlerInputKwargs(TypedDict):
    model: str
    messages: list
    optional_params: dict
    litellm_params: dict
    headers: dict
    model_response: "ModelResponse"
    logging_obj: "LiteLLMLoggingObj"
    custom_llm_provider: str


class ResponsesToCompletionBridgeHandler:
    def __init__(self):
        from .transformation import LiteLLMResponsesTransformationHandler

        super().__init__()
        self.transformation_handler = LiteLLMResponsesTransformationHandler()

    @staticmethod
    def _resolve_stream_flag(optional_params: dict, litellm_params: dict) -> bool:
        stream = optional_params.get("stream")
        if stream is None:
            stream = litellm_params.get("stream", False)
        return bool(stream)

    @staticmethod
    def _coerce_response_object(
        response_obj: Any,
        hidden_params: Optional[dict],
    ) -> "ResponsesAPIResponse":
        if isinstance(response_obj, ResponsesAPIResponse):
            response = response_obj
        elif isinstance(response_obj, dict):
            try:
                response = ResponsesAPIResponse(**response_obj)
            except Exception:
                response = ResponsesAPIResponse.model_construct(**response_obj)
        else:
            raise ValueError("Unexpected responses stream payload")

        if hidden_params:
            existing = getattr(response, "_hidden_params", None)
            if not isinstance(existing, dict) or not existing:
                setattr(response, "_hidden_params", dict(hidden_params))
            else:
                for key, value in hidden_params.items():
                    existing.setdefault(key, value)
        return response

    def _collect_response_from_stream(self, stream_iter: Any) -> "ResponsesAPIResponse":
        for _ in stream_iter:
            pass

        completed = getattr(stream_iter, "completed_response", None)
        response_obj = getattr(completed, "response", None) if completed else None
        if response_obj is None:
            raise ValueError("Stream ended without a completed response")

        hidden_params = getattr(stream_iter, "_hidden_params", None)
        response = self._coerce_response_object(response_obj, hidden_params)
        if not isinstance(response, ResponsesAPIResponse):
            raise ValueError("Stream completed response is invalid")
        return response

    async def _collect_response_from_stream_async(
        self, stream_iter: Any
    ) -> "ResponsesAPIResponse":
        async for _ in stream_iter:
            pass

        completed = getattr(stream_iter, "completed_response", None)
        response_obj = getattr(completed, "response", None) if completed else None
        if response_obj is None:
            raise ValueError("Stream ended without a completed response")

        hidden_params = getattr(stream_iter, "_hidden_params", None)
        response = self._coerce_response_object(response_obj, hidden_params)
        if not isinstance(response, ResponsesAPIResponse):
            raise ValueError("Stream completed response is invalid")
        return response

    def validate_input_kwargs(
        self, kwargs: dict
    ) -> ResponsesToCompletionBridgeHandlerInputKwargs:
        from litellm import LiteLLMLoggingObj
        from litellm.types.utils import ModelResponse

        model = kwargs.get("model")
        if model is None or not isinstance(model, str):
            raise ValueError("model is required")

        custom_llm_provider = kwargs.get("custom_llm_provider")
        if custom_llm_provider is None or not isinstance(custom_llm_provider, str):
            raise ValueError("custom_llm_provider is required")

        messages = kwargs.get("messages")
        if messages is None or not isinstance(messages, list):
            raise ValueError("messages is required")

        optional_params = kwargs.get("optional_params")
        if optional_params is None or not isinstance(optional_params, dict):
            raise ValueError("optional_params is required")

        litellm_params = kwargs.get("litellm_params")
        if litellm_params is None or not isinstance(litellm_params, dict):
            raise ValueError("litellm_params is required")

        headers = kwargs.get("headers")
        if headers is None or not isinstance(headers, dict):
            raise ValueError("headers is required")

        model_response = kwargs.get("model_response")
        if model_response is None or not isinstance(model_response, ModelResponse):
            raise ValueError("model_response is required")

        logging_obj = kwargs.get("logging_obj")
        if logging_obj is None or not isinstance(logging_obj, LiteLLMLoggingObj):
            raise ValueError("logging_obj is required")

        return ResponsesToCompletionBridgeHandlerInputKwargs(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
            model_response=model_response,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
        )

    @staticmethod
    def _mark_websearch_converted_stream_if_needed(
        logging_obj: "LiteLLMLoggingObj",
        optional_params: dict,
        litellm_params: dict,
    ) -> None:
        converted_stream = bool(
            optional_params.get("_websearch_interception_converted_stream")
            or litellm_params.get("_websearch_interception_converted_stream")
        )
        if converted_stream:
            logging_obj.model_call_details[
                "websearch_interception_converted_stream"
            ] = True

    @staticmethod
    def _call_agentic_chat_completion_hooks_sync(
        *,
        response: "ModelResponse",
        model: str,
        messages: list,
        optional_params: dict,
        logging_obj: "LiteLLMLoggingObj",
        stream: bool,
        custom_llm_provider: str,
        litellm_params: dict,
    ) -> Optional[Any]:
        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        return BaseLLMHTTPHandler()._call_agentic_chat_completion_hooks_sync(
            response=response,
            model=model,
            messages=messages,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            kwargs=litellm_params,
        )

    @staticmethod
    async def _call_agentic_chat_completion_hooks(
        *,
        response: "ModelResponse",
        model: str,
        messages: list,
        optional_params: dict,
        logging_obj: "LiteLLMLoggingObj",
        stream: bool,
        custom_llm_provider: str,
        litellm_params: dict,
    ) -> Optional[Any]:
        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        return await BaseLLMHTTPHandler()._call_agentic_chat_completion_hooks(
            response=response,
            model=model,
            messages=messages,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            kwargs=litellm_params,
        )

    @staticmethod
    def _wrap_chat_response_for_stream_if_needed(
        *,
        response: Any,
        stream: bool,
        model: str,
        logging_obj: "LiteLLMLoggingObj",
    ) -> Any:
        if not stream or not hasattr(response, "choices"):
            return response

        from litellm.llms.base_llm.base_model_iterator import (
            convert_model_response_to_streaming,
        )
        from litellm.litellm_core_utils.streaming_handler import (
            CustomStreamWrapper,
            mark_logging_obj_as_streaming,
        )

        mark_logging_obj_as_streaming(logging_obj)
        fake_stream_chunk = convert_model_response_to_streaming(response)
        return CustomStreamWrapper(
            completion_stream=iter([fake_stream_chunk]),
            model=model,
            custom_llm_provider="cached_response",
            logging_obj=logging_obj,
        )

    def _transform_response_and_run_agentic_hooks(
        self,
        *,
        model: str,
        raw_response: "ResponsesAPIResponse",
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        custom_llm_provider: str,
        encoding: Any,
        api_key: Optional[str],
        json_mode: Optional[bool],
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        stream = self._resolve_stream_flag(optional_params, litellm_params)
        response = self.transformation_handler.transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        self._mark_websearch_converted_stream_if_needed(
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        agentic_response = self._call_agentic_chat_completion_hooks_sync(
            response=response,
            model=model,
            messages=messages,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
        )
        final_response = agentic_response if agentic_response is not None else response
        return self._wrap_chat_response_for_stream_if_needed(
            response=final_response,
            stream=stream,
            model=model,
            logging_obj=logging_obj,
        )

    async def _atransform_response_and_run_agentic_hooks(
        self,
        *,
        model: str,
        raw_response: "ResponsesAPIResponse",
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        custom_llm_provider: str,
        encoding: Any,
        api_key: Optional[str],
        json_mode: Optional[bool],
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        stream = self._resolve_stream_flag(optional_params, litellm_params)
        response = self.transformation_handler.transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        self._mark_websearch_converted_stream_if_needed(
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        agentic_response = await self._call_agentic_chat_completion_hooks(
            response=response,
            model=model,
            messages=messages,
            optional_params=optional_params,
            logging_obj=logging_obj,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
        )
        final_response = agentic_response if agentic_response is not None else response
        return self._wrap_chat_response_for_stream_if_needed(
            response=final_response,
            stream=stream,
            model=model,
            logging_obj=logging_obj,
        )

    def completion(self, *args, **kwargs) -> Union[
        Coroutine[Any, Any, Union["ModelResponse", "CustomStreamWrapper"]],
        "ModelResponse",
        "CustomStreamWrapper",
    ]:
        if kwargs.get("acompletion") is True:
            return self.acompletion(**kwargs)

        from litellm import responses
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        validated_kwargs = self.validate_input_kwargs(kwargs)
        model = validated_kwargs["model"]
        messages = validated_kwargs["messages"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        if kwargs.get("stream") is not None and "stream" not in optional_params:
            optional_params = dict(optional_params)
            optional_params["stream"] = kwargs["stream"]
        headers = validated_kwargs["headers"]
        model_response = validated_kwargs["model_response"]
        logging_obj = validated_kwargs["logging_obj"]
        custom_llm_provider = validated_kwargs["custom_llm_provider"]

        request_data = self.transformation_handler.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
            litellm_logging_obj=logging_obj,
            client=kwargs.get("client"),
        )

        result = responses(
            **request_data,
        )

        stream = self._resolve_stream_flag(optional_params, litellm_params)
        if isinstance(result, ResponsesAPIResponse):
            return self._transform_response_and_run_agentic_hooks(
                model=model,
                raw_response=result,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        elif not stream:
            responses_api_response = self._collect_response_from_stream(result)
            return self._transform_response_and_run_agentic_hooks(
                model=model,
                raw_response=responses_api_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        else:
            completion_stream = self.transformation_handler.get_model_response_iterator(
                streaming_response=result,  # type: ignore
                sync_stream=True,
                json_mode=kwargs.get("json_mode"),
            )
            streamwrapper = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )
            return self._apply_post_stream_processing(
                streamwrapper, model, custom_llm_provider
            )

    async def acompletion(
        self, *args, **kwargs
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        from litellm import aresponses
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        validated_kwargs = self.validate_input_kwargs(kwargs)
        model = validated_kwargs["model"]
        messages = validated_kwargs["messages"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        if kwargs.get("stream") is not None and "stream" not in optional_params:
            optional_params = dict(optional_params)
            optional_params["stream"] = kwargs["stream"]
        headers = validated_kwargs["headers"]
        model_response = validated_kwargs["model_response"]
        logging_obj = validated_kwargs["logging_obj"]
        custom_llm_provider = validated_kwargs["custom_llm_provider"]

        try:
            request_data = self.transformation_handler.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
                litellm_logging_obj=logging_obj,
            )
        except Exception as e:
            raise e

        result = await aresponses(
            **request_data,
            aresponses=True,
        )

        stream = self._resolve_stream_flag(optional_params, litellm_params)
        if isinstance(result, ResponsesAPIResponse):
            return await self._atransform_response_and_run_agentic_hooks(
                model=model,
                raw_response=result,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        elif not stream:
            responses_api_response = await self._collect_response_from_stream_async(
                result
            )
            return await self._atransform_response_and_run_agentic_hooks(
                model=model,
                raw_response=responses_api_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        else:
            completion_stream = self.transformation_handler.get_model_response_iterator(
                streaming_response=result,  # type: ignore
                sync_stream=False,
                json_mode=kwargs.get("json_mode"),
            )
            streamwrapper = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )
            return self._apply_post_stream_processing(
                streamwrapper, model, custom_llm_provider
            )

    @staticmethod
    def _apply_post_stream_processing(
        stream: "CustomStreamWrapper",
        model: str,
        custom_llm_provider: str,
    ) -> Any:
        """Apply provider-specific post-stream processing if available."""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        try:
            provider_config = ProviderConfigManager.get_provider_chat_config(
                model=model, provider=LlmProviders(custom_llm_provider)
            )
        except (ValueError, KeyError):
            return stream

        if provider_config is not None:
            return provider_config.post_stream_processing(stream)
        return stream


responses_api_bridge = ResponsesToCompletionBridgeHandler()
