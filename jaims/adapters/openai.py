# TODO: Tokenizer history optimizer
from __future__ import annotations
import json
from enum import Enum
import time
from typing import Union
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import Stream
import tiktoken
import logging
import random
from typing import List, Optional, Dict

from ..interfaces import JAImsLLMInterface
from ..entities import (
    JAImsMessage,
    JAImsMessageContent,
    JAImsContentTypes,
    JAImsToolCall,
    JAImsToolResponse,
    JAImsFunctionTool,
    JAImsMessageRole,
)
import os

# ---------------------
# openai / LLM modeling
# ---------------------


class JAImsGPTModel(Enum):
    """
    The OPENAI GPT models available.
    """

    GPT_3_5_TURBO = ("gpt-3.5-turbo", 4096)
    GPT_3_5_TURBO_16K = ("gpt-3.5-turbo-16k", 16384)
    GPT_3_5_TURBO_0613 = ("gpt-3.5-turbo-0613", 4096)
    GPT_3_5_TURBO_16K_0613 = ("gpt-3.5-turbo-16k-0613", 16384)
    GPT_3_5_TURBO_1106 = ("gpt-3.5-turbo-1106", 16385)
    GPT_4 = ("gpt-4", 8192)
    GPT_4_32K = ("gpt-4-32k", 32768)
    GPT_4_0613 = ("gpt-4-0613", 8192)
    GPT_4_32K_0613 = ("gpt-4-32k-0613", 32768)
    GPT_4_1106_PREVIEW = ("gpt-4-1106-preview", 128000)
    GPT_4_VISION_PREVIEW = ("gpt-4-vision-preview", 128000)

    def __init__(self, string, max_tokens):
        self.string = string
        self.max_tokens = max_tokens

    def __str__(self):
        return self.string


class JAImsOpenaiKWArgs:
    """
    Represents the keyword arguments for the JAIms OpenAI wrapper.
    This class entirely mirrors the openai API parameters, so refer to it for documentation.
    (https://platform.openai.com/docs/api-reference/chat/create).

    Args:
        model (JAImsGPTModel, optional): The OpenAI model to use. Defaults to JAImsGPTModel.GPT_3_5_TURBO.
        messages (List[dict], optional): The list of messages for the chat completion. Defaults to an empty list, it is automatically populated by the run method so it is not necessary to pass them. If passed, they will always be appended to the messages passed in the run method.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 500.
        stream (bool, optional): Whether to use streaming for the API call. Defaults to False.
        temperature (float, optional): The temperature for generating creative text. Defaults to 0.0.
        top_p (Optional[int], optional): The top-p value for nucleus sampling. Defaults to None.
        n (int, optional): The number of responses to generate. Defaults to 1.
        seed (Optional[int], optional): The seed to be passed to openai to have more consistent outputs. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for avoiding repetitive responses. Defaults to 0.0.
        presence_penalty (float, optional): The presence penalty for encouraging diverse responses. Defaults to 0.0.
        logit_bias (Optional[Dict[str, float]], optional): The logit bias for influencing the model's output. Defaults to None.
        response_format (Optional[Dict], optional): The format for the generated response. Defaults to None.
        stop (Union[Optional[str], Optional[List[str]]], optional): The stop condition for the generated response. Defaults to None.
        tool_choice (Union[str, Dict], optional): The choice of tool to use. Defaults to "auto".
        tools (Optional[List[JAImsFunctionToolWrapper]], optional): The list of function tool wrappers to use. Defaults to None.
    """

    def __init__(
        self,
        model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
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
        tool_choice: Union[str, Dict] = "auto",
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
        kwargs = {
            "model": self.model.string,
            "temperature": self.temperature,
            "n": self.n,
            "stream": self.stream,
            "messages": self.messages,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "stop": self.stop,
        }

        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        if self.logit_bias:
            kwargs["logit_bias"] = self.logit_bias

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = self.tool_choice

        return kwargs

    def copy_with_overrides(
        self,
        model: Optional[JAImsGPTModel] = None,
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
    ) -> JAImsOpenaiKWArgs:
        """
        Returns a new JAImsOpenaiKWArgs instance with the passed kwargs overridden.
        """
        return JAImsOpenaiKWArgs(
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


class JAImsOptions:
    """
    Represents the options for Openai Adapter.

    Args:
        max_retries (int): The maximum number of retries after a failing openai call.
        retry_delay (int): The delay between each retry.
        exponential_base (int): The base for exponential backoff calculation.
        exponential_delay (int): The initial delay for exponential backoff.
        exponential_cap (Optional[int]): The maximum delay for exponential backoff.
        jitter (bool): Whether to add jitter to the delay (to avoid concurrent firing).
        debug_stream_function_call (bool): Prints the arguments streamed by OpenAI during function call when streaming enabled.
    """

    def __init__(
        self,
        max_retries=15,
        retry_delay=10,
        exponential_base: int = 2,
        exponential_delay: int = 1,
        exponential_cap: Optional[int] = None,
        jitter: bool = True,
        debug_stream_function_call=False,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_base = exponential_base
        self.exponential_delay = exponential_delay
        self.exponential_cap = exponential_cap
        self.jitter = jitter
        self.debug_stream_function_call = debug_stream_function_call

    def copy_with_overrides(
        self,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        exponential_base: Optional[int] = None,
        exponential_delay: Optional[int] = None,
        exponential_cap: Optional[int] = None,
        jitter: Optional[bool] = None,
        debug_stream_function_call: Optional[bool] = None,
    ) -> JAImsOptions:
        """
        Returns a new JAImsOptions instance with the passed kwargs overridden.
        """
        return JAImsOptions(
            max_retries=max_retries if max_retries else self.max_retries,
            retry_delay=retry_delay if retry_delay else self.retry_delay,
            exponential_base=(
                exponential_base if exponential_base else self.exponential_base
            ),
            exponential_delay=(
                exponential_delay if exponential_delay else self.exponential_delay
            ),
            exponential_cap=(
                exponential_cap if exponential_cap else self.exponential_cap
            ),
            jitter=jitter if jitter else self.jitter,
            debug_stream_function_call=(
                debug_stream_function_call
                if debug_stream_function_call
                else self.debug_stream_function_call
            ),
        )


def estimate_token_count(string: str, model: JAImsGPTModel) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.encoding_for_model(model.string)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class ErrorHandlingMethod(Enum):
    FAIL = "fail"
    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


def __handle_openai_error(error: openai.OpenAIError) -> ErrorHandlingMethod:
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


class JAImsOpenaiAdapter(JAImsLLMInterface):
    """
    The JAIms OpenAI adapter.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        kwargs: Optional[JAImsOpenaiKWArgs] = None,
    ):
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OpenAI API key not provided.")
        openai.api_key = openai_api_key
        self.options = options or JAImsOptions()
        self.kwargs = kwargs or JAImsOpenaiKWArgs()

    def __jaims_messages_to_openai(self, messages: List[JAImsMessage]) -> List[dict]:

        def format_contents(contents: List[JAImsMessageContent]):
            if len(contents) == 1 and contents[0].type == JAImsContentTypes.TEXT:
                return contents[0].content
            else:
                raw_contents = []
                for c in contents:

                    if c.type == JAImsContentTypes.IMAGE:
                        raw_contents.append(
                            {"type": "image_url", "image_url": c.content}
                        )
                    elif c.type == JAImsContentTypes.TEXT:
                        raw_contents.append({"type": "text", "content": c.content})
                    else:
                        raise Exception(f"Unsupported content type: {c.type}")

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
                raw_message["role"] = "assistant"
                raw_message["tool_calls"] = format_tool_calls(m.tool_calls)
                raw_messages.append(raw_message)
                continue

            raw_message = {}
            raw_message["role"] = m.role.value

            if m.contents:
                raw_message["content"] = format_contents(m.contents)

            raw_messages.append(raw_message)

        return raw_messages

    def __jaims_tools_to_openai(self, tools: List[JAImsFunctionTool]) -> List[dict]:
        raw_tools = []
        for t in tools:
            tool_raw_dict = {
                "type": "function",
                "function": {
                    "name": t.function_tool.name,
                    "description": t.function_tool.description,
                    "parameters": t.function_tool.get_jsonapi_schema(),
                },
            }
            raw_tools.append(tool_raw_dict)

        return raw_tools

    def __openai_chat_completion_to_jaims_message(
        self, completion: ChatCompletion
    ) -> JAImsMessage:
        if len(completion.choices) == 0:
            raise Exception("OpenAI returned an empty response.")

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

        content = None
        text = None
        if message.content:
            text = message.content
            content = [JAImsMessageContent(type=JAImsContentTypes.TEXT, content=text)]

        return JAImsMessage(
            role=role,
            contents=content,
            tool_calls=tool_calls,
            text=text,
            raw=message,
        )

    def call(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> JAImsMessage:
        openai_messages = self.__jaims_messages_to_openai(messages)
        openai_tools = self.__jaims_tools_to_openai(tools)
        openai_kw_args = self.kwargs.copy_with_overrides(
            messages=openai_messages,
            tools=openai_tools,
            stream=False,
        )
        response = self.get_openai_response(openai_kw_args, self.options)

        return JAImsMessage.text_message(response.choices[0].message["content"])

    def get_openai_response(
        self,
        openai_kw_args: JAImsOpenaiKWArgs,
        call_options: JAImsOptions,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        retries = 0
        logger = logging.getLogger(__name__)
        # keeps how long to sleep between retries
        sleep_time = call_options.retry_delay
        # keeps track of the exponential backoff
        backoff_time = call_options.exponential_delay

        # print(json.dumps(openai_kw_args.messages, indent=4))

        while retries < call_options.max_retries:
            try:
                client = OpenAI()
                kwargs = openai_kw_args.to_dict()
                response = client.chat.completions.create(
                    **kwargs,
                )

                return response
            except openai.OpenAIError as error:
                logger.error(f"OpenAI API error:\n{error}\n")
                error_handling_method = __handle_openai_error(error)

                if error_handling_method == ErrorHandlingMethod.FAIL:
                    raise Exception(f"OpenAI API error: {error}")

                if error_handling_method == ErrorHandlingMethod.RETRY:
                    sleep_time = call_options.retry_delay

                elif error_handling_method == ErrorHandlingMethod.EXPONENTIAL_BACKOFF:
                    logger.info(f"Performing exponential backoff")
                    jitter = 1 + call_options.jitter * random.random()
                    backoff_time = backoff_time * call_options.exponential_base * jitter

                    if (
                        call_options.exponential_cap is not None
                        and backoff_time > call_options.exponential_cap
                    ):
                        backoff_time = call_options.exponential_cap * jitter

                    sleep_time = backoff_time

                logger.warning(f"Retrying in {sleep_time} seconds")
                time.sleep(sleep_time)
                retries += 1

        max_retries_error = f"Max retries exceeded! OpenAI API call failed {call_options.max_retries} times."
        logger.error(max_retries_error)
        raise Exception(max_retries_error)
