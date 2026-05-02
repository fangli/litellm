"""
Handler for the Anthropic v1/messages -> OpenAI Responses API path.

Used when the target model is an OpenAI or Azure model.
"""

from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union

import litellm
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import ResponsesAPIResponse

from .streaming_iterator import AnthropicResponsesStreamWrapper
from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


def _build_anthropic_optional_params(
    *,
    max_tokens: int,
    context_management: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    output_config: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    output_format: Optional[Dict] = None,
) -> Dict[str, Any]:
    optional_params: Dict[str, Any] = {"max_tokens": max_tokens}
    for key, value in {
        "context_management": context_management,
        "metadata": metadata,
        "output_config": output_config,
        "stop_sequences": stop_sequences,
        "system": system,
        "temperature": temperature,
        "thinking": thinking,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_k": top_k,
        "top_p": top_p,
        "output_format": output_format,
    }.items():
        if value is not None:
            optional_params[key] = value
    return optional_params


async def _run_agentic_hooks_if_needed(
    *,
    response: AnthropicMessagesResponse,
    model: str,
    messages: List[Dict],
    stream: bool,
    anthropic_messages_optional_request_params: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    logging_obj = kwargs.get("litellm_logging_obj")
    if (
        kwargs.get("_websearch_interception_converted_stream")
        and logging_obj is not None
    ):
        logging_obj.model_call_details["websearch_interception_converted_stream"] = True

    custom_llm_provider = kwargs.get("custom_llm_provider", "")
    callback_kwargs = {**kwargs, "custom_llm_provider": custom_llm_provider}
    agentic_response = await BaseLLMHTTPHandler()._call_agentic_completion_hooks(
        response=response,
        model=model,
        messages=messages,
        anthropic_messages_provider_config=None,  # type: ignore[arg-type]
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        logging_obj=logging_obj,
        stream=stream,
        custom_llm_provider=custom_llm_provider,
        kwargs=callback_kwargs,
    )
    return agentic_response if agentic_response is not None else response


def _run_agentic_hooks_if_needed_sync(
    *,
    response: AnthropicMessagesResponse,
    model: str,
    messages: List[Dict],
    stream: bool,
    anthropic_messages_optional_request_params: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    from litellm.litellm_core_utils.asyncify import run_async_function

    return run_async_function(
        _run_agentic_hooks_if_needed,
        response=response,
        model=model,
        messages=messages,
        stream=stream,
        anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
        kwargs=kwargs,
    )


def _build_responses_kwargs(
    *,
    max_tokens: int,
    messages: List[Dict],
    model: str,
    context_management: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    output_config: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    output_format: Optional[Dict] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the kwargs dict to pass directly to litellm.responses() / litellm.aresponses().
    """
    # Build a typed AnthropicMessagesRequest for the adapter
    request_data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if context_management:
        request_data["context_management"] = context_management
    if output_config:
        request_data["output_config"] = output_config
    if metadata:
        request_data["metadata"] = metadata
    if system:
        request_data["system"] = system
    if temperature is not None:
        request_data["temperature"] = temperature
    if thinking:
        request_data["thinking"] = thinking
    if tool_choice:
        request_data["tool_choice"] = tool_choice
    if tools:
        request_data["tools"] = tools
    if top_p is not None:
        request_data["top_p"] = top_p
    if output_format:
        request_data["output_format"] = output_format

    anthropic_request = AnthropicMessagesRequest(**request_data)  # type: ignore[typeddict-item]
    responses_kwargs = _ADAPTER.translate_request(anthropic_request)

    # Normalize reasoning effort based on model capabilities
    # (e.g. "max" → "xhigh"/"high", "minimal" → "low" if unsupported)
    reasoning = responses_kwargs.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        from litellm.llms.anthropic.experimental_pass_through.utils import (
            normalize_reasoning_effort_value,
        )

        effort = reasoning["effort"]
        normalized = normalize_reasoning_effort_value(
            effort,
            model=model,
            custom_llm_provider=(extra_kwargs or {}).get("custom_llm_provider"),
        )
        if normalized != effort:
            responses_kwargs["reasoning"] = {**reasoning, "effort": normalized}

    if stream:
        responses_kwargs["stream"] = True

    # Forward litellm-specific kwargs (api_key, api_base, logging obj, etc.)
    excluded = {"anthropic_messages"}
    for key, value in (extra_kwargs or {}).items():
        if key == "litellm_logging_obj" and value is not None:
            from litellm.litellm_core_utils.litellm_logging import (
                Logging as LiteLLMLoggingObject,
            )
            from litellm.types.utils import CallTypes

            if isinstance(value, LiteLLMLoggingObject):
                setattr(value, "call_type", CallTypes.responses.value)
            responses_kwargs[key] = value
        elif key not in excluded and key not in responses_kwargs and value is not None:
            responses_kwargs[key] = value

    return responses_kwargs


class LiteLLMMessagesToResponsesAPIHandler:
    """
    Handles Anthropic /v1/messages requests for OpenAI / Azure models by
    calling litellm.responses() / litellm.aresponses() directly and translating
    the response back to Anthropic format.
    """

    @staticmethod
    async def async_anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        result = await litellm.aresponses(**responses_kwargs)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        translated_response = _ADAPTER.translate_response(result)
        return await _run_agentic_hooks_if_needed(
            response=translated_response,
            model=model,
            messages=messages,
            stream=bool(stream),
            anthropic_messages_optional_request_params=_build_anthropic_optional_params(
                max_tokens=max_tokens,
                context_management=context_management,
                metadata=metadata,
                output_config=output_config,
                stop_sequences=stop_sequences,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
            ),
            kwargs=kwargs,
        )

    @staticmethod
    def anthropic_messages_handler(
        max_tokens: int,
        messages: List[Dict],
        model: str,
        context_management: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        output_config: Optional[Dict] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Dict] = None,
        tool_choice: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_format: Optional[Dict] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> Union[
        AnthropicMessagesResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any]]],
    ]:
        if _is_async:
            return (
                LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    context_management=context_management,
                    metadata=metadata,
                    output_config=output_config,
                    stop_sequences=stop_sequences,
                    stream=stream,
                    system=system,
                    temperature=temperature,
                    thinking=thinking,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_k=top_k,
                    top_p=top_p,
                    output_format=output_format,
                    **kwargs,
                )
            )

        # Sync path
        responses_kwargs = _build_responses_kwargs(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            context_management=context_management,
            metadata=metadata,
            output_config=output_config,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            output_format=output_format,
            extra_kwargs=kwargs,
        )

        result = litellm.responses(**responses_kwargs)

        if stream:
            wrapper = AnthropicResponsesStreamWrapper(
                responses_stream=result, model=model
            )
            return wrapper.async_anthropic_sse_wrapper()

        if not isinstance(result, ResponsesAPIResponse):
            raise ValueError(f"Expected ResponsesAPIResponse, got {type(result)}")

        translated_response = _ADAPTER.translate_response(result)
        return _run_agentic_hooks_if_needed_sync(
            response=translated_response,
            model=model,
            messages=messages,
            stream=bool(stream),
            anthropic_messages_optional_request_params=_build_anthropic_optional_params(
                max_tokens=max_tokens,
                context_management=context_management,
                metadata=metadata,
                output_config=output_config,
                stop_sequences=stop_sequences,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                tools=tools,
                top_k=top_k,
                top_p=top_p,
                output_format=output_format,
            ),
            kwargs=kwargs,
        )
