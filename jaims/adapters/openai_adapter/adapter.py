from __future__ import annotations
import json
from math import ceil
from typing import Generator, Union
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
from openai import Stream
import tiktoken
from typing import List, Optional, Dict, Literal
from PIL import Image

from ...interfaces import (
    LLMAdapterITF,
    HistoryOptimizerITF,
)
from ...entities import (
    ImageContent,
    ContentType,
    Message,
    StreamingMessage,
    ToolCall,
    FunctionTool,
    MessageRole,
    Config,
)
from ..shared.image_utilities import image_to_b64
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

import os
from copy import deepcopy

# ---------------------
# openai / LLM modeling
# ---------------------


class OpenAIParams:
    """
    Represents the keyword arguments for the JAIms OpenAI wrapper.
    This class entirely mirrors the openai API parameters, so refer to it for documentation.
    (https://platform.openai.com/docs/api-reference/chat/create).

    Args:
        model (str, optional): The ID of the model to use. Defaults to "gpt-4o".
        messages (List[dict], optional): A list of messages. Defaults to [].
        max_tokens (int, optional): The maximum number of tokens allowed. Defaults to 1024.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        temperature (float, optional): What sampling temperature to use. Defaults to 0.0.
        top_p (Optional[int], optional): The nucleus sampling probability. Defaults to None.
        n (int, optional): How many completions to generate. Defaults to 1. (Currently only 1 is supported in JAIms).
        seed (Optional[int], optional): The seed to use for the model. Defaults to None.
        frequency_penalty (float, optional): How much to penalize new tokens based on their frequency in the text so far. Defaults to 0.0.
        presence_penalty (float, optional): How much to penalize new tokens based on whether they appear in the text so far. Defaults to 0.0.
        logit_bias (Optional[Dict[str, float]], optional): A map of tokens to their logit bias values. Defaults to None.
        response_format (Optional[Dict], optional): The format of the response. Defaults to None.
        stop (Union[Optional[str], Optional[List[str]]], optional): One or more sequences where the API will stop generating tokens. Defaults to None.
        tool_choice (Optional[Union[str, Dict]], optional): The tool choice. Defaults to None.
        tools (Optional[List[Dict]], optional): The tools json schema. Defaults to None.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        messages: List[dict] = [],
        max_tokens: int = 1024,
        stream: bool = False,
        temperature: float = 0.0,
        top_p: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        response_format: Optional[Dict] = None,
        stop: Union[Optional[str], Optional[List[str]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        tools: Optional[List[Dict]] = None,
    ):
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias
        self.response_format = response_format
        self.stop = stop
        self.tool_choice = tool_choice
        self.tools = tools

    def to_dict(self):
        """
        Returns the llm parameters as a dictionary, removing None values.

        Returns:
            dict: The llm parameters as a dictionary.
        """

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "n": self.n,
            "stream": self.stream,
            "messages": self.messages,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "stop": self.stop,
        }

        # Remove None values
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        return kwargs

    @staticmethod
    def from_dict(raw_dictionary: dict) -> OpenAIParams:
        """
        Returns a new OpenAIParams instance from a dictionary.

        Args:
            kwargs (dict): The dictionary containing the parameters.

        Returns:
            OpenAIParams: The OpenAIParams instance.
        """
        return OpenAIParams(
            model=raw_dictionary.get("model", "gpt-4o"),
            messages=raw_dictionary.get("messages", []),
            max_tokens=raw_dictionary.get("max_tokens", 1024),
            stream=raw_dictionary.get("stream", False),
            temperature=raw_dictionary.get("temperature", 0.0),
            top_p=raw_dictionary.get("top_p", None),
            n=raw_dictionary.get("n", 1),
            seed=raw_dictionary.get("seed", None),
            frequency_penalty=raw_dictionary.get("frequency_penalty", 0.0),
            presence_penalty=raw_dictionary.get("presence_penalty", 0.0),
            logit_bias=raw_dictionary.get("logit_bias", None),
            response_format=raw_dictionary.get("response_format", None),
            stop=raw_dictionary.get("stop", None),
            tool_choice=raw_dictionary.get("tool_choice", None),
            tools=raw_dictionary.get("tools", None),
        )

    def copy_with_overrides(
        self,
        model: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        response_format: Optional[Dict] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        tools: Optional[List[Dict]] = None,
    ) -> OpenAIParams:
        """
        Returns a new OpenAIParams instance with the provided parameters overridden. Pass only the parameters you want to override.

        Args:
            model (Optional[str], optional): The ID of the model to use. Defaults to None.
            messages (Optional[List[dict]], optional): A list of messages. Defaults to None.
            max_tokens (Optional[int], optional): The maximum number of tokens allowed. Defaults to None.
            stream (Optional[bool], optional): Whether to stream the response. Defaults to None.
            temperature (Optional[float], optional): What sampling temperature to use. Defaults to None.
            top_p (Optional[int], optional): The nucleus sampling probability. Defaults to None.
            n (Optional[int], optional): How many completions to generate. Defaults to None.
            seed (Optional[int], optional): The seed to use for the model. Defaults to None.
            frequency_penalty (Optional[float], optional): How much to penalize new tokens based on their frequency in the text so far. Defaults to None.
            presence_penalty (Optional[float], optional): How much to penalize new tokens based on whether they appear in the text so far. Defaults to None.
            logit_bias (Optional[Dict[str, float]], optional): A map of tokens to their logit bias values. Defaults to None.
            response_format (Optional[Dict], optional): The format of the response. Defaults to None.
            stop (Optional[Union[str, List[str]], optional): One or more sequences where the API will stop generating tokens. Defaults to None.
            tool_choice (Optional[Union[str, Dict]], optional): The tool choice. Defaults to None.
            tools (Optional[List[Dict]], optional): The tools json schema. Defaults to None.

        Returns:
            OpenAIParams: The new OpenAIParams instance with the provided parameters overridden.
        """
        return OpenAIParams(
            model=model if model else self.model,
            messages=messages if messages else self.messages,
            max_tokens=max_tokens if max_tokens else self.max_tokens,
            stream=stream if stream else self.stream,
            temperature=temperature if temperature else self.temperature,
            top_p=top_p if top_p else self.top_p,
            n=n if n else self.n,
            seed=seed if seed else self.seed,
            frequency_penalty=(
                frequency_penalty if frequency_penalty else self.frequency_penalty
            ),
            presence_penalty=(
                presence_penalty if presence_penalty else self.presence_penalty
            ),
            logit_bias=logit_bias if logit_bias else self.logit_bias,
            response_format=(
                response_format if response_format else self.response_format
            ),
            stop=stop if stop else self.stop,
            tool_choice=tool_choice if tool_choice else self.tool_choice,
            tools=tools if tools else self.tools,
        )


class OpenAITokenHistoryOptimizer(HistoryOptimizerITF):
    def __init__(
        self,
        config: Config,
        history_max_tokens: int,
        model: str,
    ):
        self.config = config
        self.history_max_tokens = history_max_tokens
        self.model = model

    def optimize_history(self, messages: List[Message]) -> List[Message]:
        """
        Optimizes the chat history by removing the oldest messages until the total number of tokens is less than or equal to the maximum allowed tokens.

        Args:
            messages (List[Message]): The chat history between the user and agent.

        Returns:
            List: The optimized chat history.
        """

        # Copying the whole history to avoid altering the original one
        buffer = messages.copy()

        # calculate the tokens for the compound history
        messages_tokens = self.__tokens_from_messages(buffer, self.model)

        while messages_tokens > self.history_max_tokens:
            if not buffer:
                raise Exception(
                    f"Unable to fit messages with current max tokens {self.history_max_tokens}."
                )
            # Popping the first (oldest) message from the chat history between the user and agent
            buffer.pop(0)
            # Recalculating the tokens for the compound history
            messages_tokens = self.__tokens_from_messages(buffer, self.model)

        return buffer

    def __estimate_token_count(self, string: str, model: str) -> int:
        """Returns the number of tokens in a text string."""

        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def __estimate_image_tokens_count(self, width: int, height: int):
        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n
        return total

    def __tokens_from_messages(self, messages: List[Message], model):
        """Returns the number of tokens used by a list of messages."""

        images = []
        parsed = []
        for message in messages:
            if message.contents:
                for item in message.contents:
                    if isinstance(item, str):
                        parsed.append(item)
                    elif isinstance(item, Image.Image):
                        images.append(item)
                    else:
                        raise Exception(f"Unsupported content type: {type(item)}")

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parsed.append(tool_call.tool_name + json.dumps(tool_call.tool_args))

            if message.tool_responses:
                for tool_response in message.tool_responses:
                    parsed.append(
                        tool_response.tool_name
                        + tool_response.tool_call_id
                        + json.dumps(tool_response.response),
                    )

        image_tokens = 0
        for image in images:
            width, height = image.size
            image_tokens += self.__estimate_image_tokens_count(width, height)

        return self.__estimate_token_count(json.dumps(parsed), model) + image_tokens


class OpenaiAdapter(LLMAdapterITF):
    """
    OpenAI adapter for JAIms.

    Args:
        api_key (Optional[str], optional): The OpenAI API key. Defaults to None.
        options (Optional[Config], optional): The configuration options for the adapter. Defaults to None.
        kwargs (Optional[Union[OpenAIParams, Dict]], optional): The keyword arguments for the OpenAI API. Defaults to None.
        existing_params_messages_behaviour (Literal["append", "replace"], optional): The behavior for the messages in the kwargs when receiving new messages on calls. Defaults to "append", which appends new messages to the params passed in the constructor.
        existing_tools_behaviour (Literal["append", "replace"], optional): The behavior for the tools in the kwargs when receiving new tools on calls. Defaults to "append", which appends new tools to the params passed in the constructor.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        options: Optional[Config] = None,
        kwargs: Optional[Union[OpenAIParams, Dict]] = None,
        existing_params_messages_behaviour: Literal["append", "replace"] = "append",
        existing_tools_behaviour: Literal["append", "replace"] = "append",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise Exception("OpenAI API key not provided.")

        self.options = options or Config()
        self.kwargs = kwargs or OpenAIParams()
        self.existing_params_messages_behaviour = existing_params_messages_behaviour
        self.existing_tools_behaviour = existing_tools_behaviour

    def __get_args(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
        stream: bool = False,
    ):

        if tool_constraints and len(tool_constraints) > 1:
            raise ValueError(
                "Only one tool choice is allowed when using the OpenAI API."
            )

        if isinstance(self.kwargs, OpenAIParams):
            args = self.kwargs.to_dict()
        else:
            args = deepcopy(self.kwargs)

        # handle messages

        openai_messages = self.__jaims_messages_to_openai(messages or [])
        if self.existing_params_messages_behaviour == "append":
            kwargs_messages = args.get("messages", [])
            openai_messages = kwargs_messages + openai_messages

        args["messages"] = openai_messages
        args["stream"] = stream

        # handle tools

        openai_tools = self.__jaims_tools_to_openai(tools or [])
        if len(openai_tools) > 0:
            if self.existing_tools_behaviour == "append":
                openai_tools = args.get("tools", []) + openai_tools

            tool_choice = "auto"
            if tool_constraints:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": tool_constraints[0],
                    },
                }
            elif args.get("tool_choice", None):
                tool_choice = args["tool_choice"]

            args["tools"] = openai_tools
            args["tool_choice"] = tool_choice

        return args

    def call(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Message:
        args = self.__get_args(messages, tools, tool_constraints)
        response = self.___get_openai_response(args, self.options)
        assert isinstance(response, ChatCompletion)

        return self.__openai_chat_completion_to_jaims_message(response)

    def call_streaming(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[StreamingMessage, None, None]:
        args = self.__get_args(
            messages, tools, stream=True, tool_constraints=tool_constraints
        )
        response = self.___get_openai_response(args, self.options)
        assert isinstance(response, Stream)

        accumulated_delta = None
        for completion_chunk in response:
            accumulated_delta = self.__accumulate_choice_delta(
                accumulated_delta, completion_chunk.choices[0].delta
            )
            yield self.__openai_chat_completion_choice_delta_to_jaims_message(
                accumulated_delta, completion_chunk
            )

    def __jaims_messages_to_openai(self, messages: List[Message]) -> List[dict]:

        def format_contents(contents: List[ContentType]):
            if len(contents) == 1 and isinstance(contents[0], str):
                return contents[0]
            else:
                raw_contents = []
                for c in contents:
                    if isinstance(c, str):
                        raw_contents.append({"type": "text", "text": c})
                    elif isinstance(c, ImageContent):
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

        def format_tool_calls(tool_calls: List[ToolCall]):
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

    def __jaims_tools_to_openai(self, tools: List[FunctionTool]) -> List[dict]:
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

    def __openai_chat_completion_to_jaims_message(
        self, completion: ChatCompletion
    ) -> Message:
        if len(completion.choices) == 0:
            raise Exception("OpenAI returned an empty response.")

        message = completion.choices[0].message
        role = MessageRole(message.role)
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    tool_args=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        return Message(
            role=role,
            contents=[message.content] if message.content else None,
            tool_calls=tool_calls,
            raw=message,
        )

    def __openai_chat_completion_choice_delta_to_jaims_message(
        self, accumulated_choice_delta: ChoiceDelta, current_chunk: ChatCompletionChunk
    ) -> StreamingMessage:

        current_choice = current_chunk.choices[0]
        role = (
            MessageRole(current_choice.delta.role)
            if current_choice.delta.role
            else accumulated_choice_delta.role
        )
        textDelta = current_choice.delta.content
        contents: Optional[List[ContentType]] = None
        function_tool_calls = None

        role = MessageRole(accumulated_choice_delta.role)
        if accumulated_choice_delta.content:
            contents = [
                accumulated_choice_delta.content,
            ]

        if current_choice.finish_reason and accumulated_choice_delta.tool_calls:
            function_tool_calls = []
            for tc in accumulated_choice_delta.tool_calls:
                if tc.function:
                    function_tool_calls.append(
                        ToolCall(
                            id=tc.id or "",
                            tool_name=tc.function.name or "",
                            tool_args=(
                                json.loads(tc.function.arguments)
                                if tc.function.arguments
                                else None
                            ),
                        )
                    )

        return StreamingMessage(
            message=Message(
                role=role,
                contents=contents,
                tool_calls=function_tool_calls,
                raw=accumulated_choice_delta,
            ),
            textDelta=textDelta,
        )

    def ___get_openai_response(
        self,
        openai_kw_args: dict,
        call_options: Config,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:

        def handle_openai_error(error) -> ErrorHandlingMethod:
            # errors are handled according to the guidelines here: https://platform.openai.com/docs/guides/error-codes/api-errors (dated 03/10/2023)
            # this map indexes all the error that require a retry or an exponential backoff, every other error is a fail
            error_handling_map = {
                openai.RateLimitError: ErrorHandlingMethod.EXPONENTIAL_BACKOFF,
                openai.InternalServerError: ErrorHandlingMethod.RETRY,
                openai.APITimeoutError: ErrorHandlingMethod.RETRY,
            }

            for error_type, error_handling_method in error_handling_map.items():
                if isinstance(error, error_type):
                    return error_handling_method

            return ErrorHandlingMethod.FAIL

        def openai_api_call():
            client = OpenAI(api_key=self.api_key)
            kwargs = openai_kw_args
            response = client.chat.completions.create(
                **kwargs,
            )

            return response

        return call_with_exponential_backoff(
            openai_api_call,
            handle_openai_error,
            call_options,
        )

    def __merge_tool_calls(
        self,
        existing_tool_calls: Optional[List[ChoiceDeltaToolCall]],
        new_tool_calls_delta: List[ChoiceDeltaToolCall],
    ):
        if not existing_tool_calls:
            return new_tool_calls_delta

        new_tool_calls = existing_tool_calls[:]
        for new_call_delta in new_tool_calls_delta:
            # check the tall call is already being streamed
            existing_call = next(
                (item for item in new_tool_calls if item.index == new_call_delta.index),
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
        self, accumulator: Optional[ChoiceDelta], new_delta: ChoiceDelta
    ) -> ChoiceDelta:
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
