from __future__ import annotations
from abc import ABC, abstractmethod
import json
from typing import Generator, Union
from mistralai.exceptions import MistralAPIStatusException, MistralConnectionException
from mistralai.client import MistralClient
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ToolCall,
)
from typing import List, Optional, Dict, Literal
from PIL import Image

from ...interfaces import JAImsLLMInterface
from ...entities import (
    JAImsImageContent,
    JAImsContentType,
    JAImsMessage,
    JAImsStreamingMessage,
    JAImsToolCall,
    JAImsFunctionTool,
    JAImsMessageRole,
    JAImsOptions,
)
from ..shared.image_utilities import image_to_b64
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

import os
from copy import deepcopy

# ---------------------
# Mistral / LLM modeling
# ---------------------


class JAImsMistralKWArgs:
    """
    Represents the keyword arguments for the JAIms Mistral wrapper.
    This class entirely mirrors the Mistral API parameters, so refer to it for documentation.
    (https://docs.mistral.ai/api/#operation/createChatCompletion).

    Args:
        model (str, optional): The Mistral model to use. Defaults to open-mistral-7b.
        messages (List[dict], optional): The list of messages for the chat completion. Defaults to an empty list, it is automatically populated by the run method so it is not necessary to pass them. If passed, they will always be appended to the messages passed in the run method.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 500.
        stream (bool, optional): Whether to use streaming for the API call. Defaults to False.
        temperature (float, optional): The temperature for generating creative text. Defaults to 0.0.
        top_p (Optional[int], optional): The top-p value for nucleus sampling. Defaults to None.
        response_format (Optional[Dict], optional): The format for the generated response. Defaults to None.
        stop (Union[Optional[str], Optional[List[str]]], optional): The stop condition for the generated response. Defaults to None.
        tool_choice ([Literal["auto", "any", "none"], optional): The choice of tool to use. Defaults to "auto".
        tools (Optional[List[JAImsFunctionToolWrapper]], optional): The list of function tool wrappers to use. Defaults to None.
    """

    def __init__(
        self,
        model: str = "open-mistral-7b",
        messages: List[dict] = [],
        max_tokens: int = 1024,
        stream: bool = False,
        temperature: float = 0.0,
        top_p: Optional[int] = None,
        random_seed: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stop: Union[Optional[str], Optional[List[str]]] = None,
        tool_choice: Optional[Literal["auto", "any", "none"]] = None,
        tools: Optional[List[Dict]] = None,
    ):
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.seed = random_seed
        self.response_format = response_format
        self.stop = stop
        self.tool_choice = tool_choice
        self.tools = tools

    def to_dict(self):
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
            "messages": self.messages,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "response_format": self.response_format,
            "stop": self.stop,
        }

        # Remove None values
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        return kwargs

    @staticmethod
    def from_dict(kwargs: dict) -> JAImsMistralKWArgs:
        return JAImsMistralKWArgs(
            model=kwargs.get("model", "open-mistral-7b"),
            messages=kwargs.get("messages", []),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=kwargs.get("stream", False),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", None),
            stop=kwargs.get("stop", None),
            tool_choice=kwargs.get("tool_choice", None),
            response_format=kwargs.get("response_format", None),
            tools=kwargs.get("tools", None),
        )

    def copy_with_overrides(
        self,
        model: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tool_choice: Optional[Literal["auto", "any", "none"]] = None,
        tools: Optional[List[Dict]] = None,
    ) -> JAImsMistralKWArgs:
        """
        Returns a new JAImsMistralKWArgs instance with the passed kwargs overridden.
        """
        return JAImsMistralKWArgs(
            model=model if model else self.model,
            messages=messages if messages else self.messages,
            max_tokens=max_tokens if max_tokens else self.max_tokens,
            stream=stream if stream else self.stream,
            temperature=temperature if temperature else self.temperature,
            top_p=top_p if top_p else self.top_p,
            response_format=(
                response_format if response_format else self.response_format
            ),
            stop=stop if stop else self.stop,
            tool_choice=tool_choice,
            tools=tools if tools else self.tools,
        )


class MistralTransactionStorageInterface(ABC):
    """
    Interface for storing LLM transactions.
    Override this class to implement your own storage, to store a pair of LLM request and response payloads.
    """

    @abstractmethod
    def store_transaction(self, request: dict, response: dict):
        pass


class JAImsMistralAdapter(JAImsLLMInterface):
    """
    The JAIms Mistral adapter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        kwargs: Optional[Union[JAImsMistralKWArgs, Dict]] = None,
        kwargs_messages_behavior: Literal["append", "replace"] = "append",
        kwargs_tools_behavior: Literal["append", "replace"] = "append",
        transaction_storage: Optional[MistralTransactionStorageInterface] = None,
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise Exception("Mistral API key not provided.")

        self.options = options or JAImsOptions()
        self.kwargs = kwargs or JAImsMistralKWArgs()
        self.kwargs_messages_behavior = kwargs_messages_behavior
        self.kwargs_tools_behavior = kwargs_tools_behavior
        self.transaction_storage = transaction_storage

    def __get_args(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
        stream: bool = False,
    ):
        if isinstance(self.kwargs, JAImsMistralKWArgs):
            args = self.kwargs.to_dict()
        else:
            args = deepcopy(self.kwargs)

        mistral_messages = self.__jaims_messages_to_mistral(messages or [])
        if self.kwargs_messages_behavior == "append":
            kwargs_messages = args.get("messages", [])
            mistral_messages = kwargs_messages + mistral_messages

        args["messages"] = mistral_messages
        args["stream"] = stream

        # handle tools

        mistral_tools = self.__jaims_tools_to_mistral(tools or [])
        if self.kwargs_tools_behavior == "append":
            mistral_tools = args.get("tools", []) + mistral_tools

        tool_choice = "auto"
        if tool_constraints is not None:
            if len(tool_constraints) >= 1:
                tool_choice = "any"
            else:
                tool_choice = "none"

        elif args.get("tool_choice"):
            tool_choice = args.get("tool_choice")

        if mistral_tools:
            args["tools"] = mistral_tools
            args["tool_choice"] = tool_choice

        return args

    def call(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> JAImsMessage:
        args = self.__get_args(
            messages=messages,
            tools=tools,
            tool_constraints=tool_constraints,
        )
        response = self.___get_mistral_response(args, self.options)
        assert isinstance(response, ChatCompletionResponse)
        if self.transaction_storage:
            self.transaction_storage.store_transaction(
                request=args,
                response=response.model_dump(exclude_none=True),
            )

        return self.__mistral_chat_completion_to_jaims_message(response)

    def call_streaming(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[JAImsStreamingMessage, None, None]:
        args = self.__get_args(
            messages, tools, stream=True, tool_constraints=tool_constraints
        )
        response = self.___get_mistral_response(args, self.options)

        accumulated_delta = None
        for completion_chunk in response:
            assert isinstance(completion_chunk, ChatCompletionStreamResponse)
            accumulated_delta = self.__accumulate_choice_delta(
                accumulated_delta, completion_chunk.choices[0].delta
            )
            yield self.__mistral_chat_completion_choice_delta_to_jaims_message(
                accumulated_delta, completion_chunk
            )

        if self.transaction_storage and accumulated_delta:
            self.transaction_storage.store_transaction(
                request=args,
                response=accumulated_delta.model_dump(exclude_none=True),
            )

    def __jaims_messages_to_mistral(self, messages: List[JAImsMessage]) -> List[dict]:
        def format_contents(contents: List[JAImsContentType]):
            if len(contents) == 1 and isinstance(contents[0], str):
                return contents[0]
            else:
                raw_contents = []
                for c in contents:
                    if isinstance(c, str):
                        raw_contents.append({"type": "text", "text": c})
                    elif isinstance(c, JAImsImageContent):
                        url = c.image
                        if isinstance(c.image, Image.Image):
                            mime, b64 = image_to_b64(c.image)
                            url = f"data:{mime};base64,{b64}"

                        raw_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            }
                        )

                    else:
                        raise Exception(f"Unsupported content type: {type(c)}")

                return raw_contents

        def format_tool_calls(tool_calls: List[JAImsToolCall]):
            raw_tool_calls = []
            for tc in tool_calls:
                raw_tool_call = {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.tool_args),
                    },
                }
                raw_tool_calls.append(raw_tool_call)
            return raw_tool_calls

        raw_messages = []
        for m in messages:
            if m.tool_responses:
                for tr in m.tool_responses:
                    raw_messages.append(
                        {
                            "role": "tool",
                            "name": tr.tool_name,
                            "tool_call_id": tr.tool_call_id,
                            "content": json.dumps(tr.response),
                        }
                    )
                continue

            if m.tool_calls:
                raw_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": format_tool_calls(m.tool_calls),
                    }
                )
                continue

            raw_message = {}
            raw_message["role"] = m.role.value

            if m.contents:
                raw_message["content"] = format_contents(m.contents)

            raw_messages.append(raw_message)

        return raw_messages

    def __jaims_tools_to_mistral(self, tools: List[JAImsFunctionTool]) -> List[dict]:
        raw_tools = []
        for t in tools:
            tool_raw_dict = {
                "type": "function",
                "function": {
                    "name": t.descriptor.name,
                    "description": t.descriptor.description,
                    "parameters": t.descriptor.json_schema(),
                },
            }
            raw_tools.append(tool_raw_dict)

        return raw_tools

    def __mistral_chat_completion_to_jaims_message(
        self, completion: ChatCompletionResponse
    ) -> JAImsMessage:
        if len(completion.choices) == 0:
            raise Exception("Mistral returned an empty response.")

        message = completion.choices[0].message
        role = JAImsMessageRole(message.role)
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                JAImsToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    tool_args=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        return JAImsMessage(
            role=role,
            contents=[message.content] if message.content else None,
            tool_calls=tool_calls,
            raw=message,
        )

    def __mistral_chat_completion_choice_delta_to_jaims_message(
        self,
        accumulated_choice_delta: DeltaMessage,
        current_chunk: ChatCompletionStreamResponse,
    ) -> JAImsStreamingMessage:
        current_choice = current_chunk.choices[0]
        role = (
            JAImsMessageRole(current_choice.delta.role)
            if current_choice.delta.role
            else accumulated_choice_delta.role
        )
        textDelta = current_choice.delta.content
        contents: Optional[List[JAImsContentType]] = None
        function_tool_calls = None

        role = JAImsMessageRole(accumulated_choice_delta.role)
        if accumulated_choice_delta.content:
            contents = [
                accumulated_choice_delta.content,
            ]

        if current_choice.finish_reason and accumulated_choice_delta.tool_calls:
            function_tool_calls = []
            for tc in accumulated_choice_delta.tool_calls:
                if tc.function:
                    function_tool_calls.append(
                        JAImsToolCall(
                            id=tc.id or "",
                            tool_name=tc.function.name or "",
                            tool_args=(
                                json.loads(tc.function.arguments)
                                if tc.function.arguments
                                else None
                            ),
                        )
                    )

        return JAImsStreamingMessage(
            message=JAImsMessage(
                role=role,
                contents=contents,
                tool_calls=function_tool_calls,
                raw=accumulated_choice_delta,
            ),
            textDelta=textDelta,
        )

    def ___get_mistral_response(
        self,
        mistral_kw_args: dict,
        call_options: JAImsOptions,
    ) -> Union[ChatCompletionResponse, ChatCompletionStreamResponse]:
        kwargs = mistral_kw_args.copy()
        is_stream = kwargs.pop("stream", False)

        def handle_mistral_error(error) -> ErrorHandlingMethod:
            # this map indexes all the error that require a retry or an exponential backoff, every other error is a fail
            error_handling_map = {
                MistralConnectionException: ErrorHandlingMethod.RETRY,
                MistralAPIStatusException: ErrorHandlingMethod.RETRY,
            }

            for error_type, error_handling_method in error_handling_map.items():
                if isinstance(error, error_type):
                    return error_handling_method

            return ErrorHandlingMethod.FAIL

        def mistral_api_call():
            client = MistralClient(api_key=self.api_key)

            if is_stream:
                response = client.chat_stream(**kwargs)
            else:
                response = client.chat(**kwargs)

            return response

        return call_with_exponential_backoff(
            mistral_api_call,
            handle_mistral_error,
            call_options,
        )

    def __merge_tool_calls(
        self,
        existing_tool_calls: Optional[List[ToolCall]],
        new_tool_calls_delta: List[ToolCall],
    ):
        if not existing_tool_calls:
            return new_tool_calls_delta

        new_tool_calls = existing_tool_calls[:]
        for new_call_delta in new_tool_calls_delta:
            # check the tall call is already being streamed
            existing_call = next(
                (item for item in new_tool_calls if item.id == new_call_delta.id),
                None,
            )
            # new tool call, add it to the list
            if not existing_call:
                new_tool_calls.append(new_call_delta)

            # existing tool call, update it
            else:
                # update tool type
                if (
                    existing_call.type != new_call_delta.type
                    and new_call_delta.type is not None
                ):
                    existing_call.type = new_call_delta.type

                # update tool id
                if (
                    existing_call.id != new_call_delta.id
                    and new_call_delta.id is not None
                ):
                    existing_call.id = new_call_delta.id

                # update function
                if new_call_delta.function:
                    if existing_call.function is None:
                        existing_call.function = new_call_delta.function
                    else:
                        # update function name
                        if (
                            existing_call.function.name != new_call_delta.function.name
                            and new_call_delta.function.name is not None
                        ):
                            existing_call.function.name = new_call_delta.function.name

                        # update function args
                        existing_call.function.arguments = (
                            existing_call.function.arguments or ""
                        ) + (new_call_delta.function.arguments or "")

        return new_tool_calls

    def __accumulate_choice_delta(
        self,
        accumulator: Optional[DeltaMessage],
        new_delta: DeltaMessage,
    ) -> DeltaMessage:
        if accumulator is None:
            return new_delta

        if new_delta.content:
            accumulator.content = (accumulator.content or "") + new_delta.content
        if new_delta.role:
            accumulator.role = new_delta.role
        if new_delta.tool_calls:
            accumulator.tool_calls = self.__merge_tool_calls(
                accumulator.tool_calls, new_delta.tool_calls
            )

        return accumulator
