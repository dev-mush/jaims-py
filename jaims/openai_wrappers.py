from __future__ import annotations

from enum import Enum
import time
from typing import Any, List, Optional, Dict, Union
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import Stream
import tiktoken
import logging
import random

from jaims.function_handler import JAImsFuncWrapper, parse_function_wrappers_to_tools

DEFAULT_MAX_TOKENS = 1024
MAX_CONSECUTIVE_CALLS = 10


class JAImsGPTModel(Enum):
    """
    The OPENAI GPT models available.
    Only those that support functions are listed, so just:
    gpt-3.5-turbo-0613, gpt-3-5-turbo-16k-0613, gpt-4-0613
    """

    GPT_3_5_TURBO = ("gpt-3.5-turbo", 4096, 0.0015, 0.002)
    GPT_3_5_TURBO_16K = ("gpt-3.5-turbo-16k", 16384, 0.003, 0.004)
    GPT_3_5_TURBO_0613 = ("gpt-3.5-turbo-0613", 4096, 0.0015, 0.002)
    GPT_3_5_TURBO_16K_0613 = ("gpt-3.5-turbo-16k-0613", 16384, 0.003, 0.004)
    GPT_3_5_TURBO_1106 = ("gpt-3.5-turbo-1106", 16385, 0.001, 0.002)
    GPT_4 = ("gpt-4", 8192, 0.03, 0.06)
    GPT_4_32K = ("gpt-4-32k", 32768, 0.06, 0.12)
    GPT_4_0613 = ("gpt-4-0613", 8192, 0.03, 0.06)
    GPT_4_32K_0613 = ("gpt-4-32k-0613", 32768, 0.06, 0.12)
    GPT_4_1106_PREVIEW = ("gpt-4-1106-preview", 128000, 0.01, 0.03)
    GPT_4_0125_PREVIEW = ("gpt-4-0125-preview", 128000, 0.01, 0.03)
    GPT_4_TURBO = ("gpt-4-turbo", 128000, 0.0, 0.0)
    GPT_4_TURBO_PREVIEW = ("gpt-4-turbo-preview", 128000, 0.0, 0.0)
    GPT_4_o = ("gpt-4o", 1280000, 0.00, 0.0)
    GPT_4_o_2024_05_13 = ("gpt-4o-2024-05-13", 1280000, 0.00, 0.0)
    GPT_4_TURBO_2024_04_09 = ("gpt-4-turbo-2024-04-09", 128000, 0.01, 0.03)
    GPT_4_1106_VISION_PREVIEW = ("gpt-4-1106-vision-preview", 128000, 0.01, 0.03)
    GPT_4_VISION_PREVIEW = ("gpt-4-vision-preview", 128000, 0.01, 0.03)

    def __init__(self, string, max_tokens, price_1k_tokens_in, price_1k_tokens_out):
        self.string = string
        self.max_tokens = max_tokens
        self.price_1k_tokens_in = price_1k_tokens_in
        self.price_1k_tokens_out = price_1k_tokens_out

    def __str__(self):
        return self.string


class JAImsTokensExpense:
    """
    Tracks the number of tokens spent on a job and on which GPTModel.
    """

    def __init__(
        self,
        gpt_model: JAImsGPTModel,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        rough_estimate=False,
    ):
        self.gpt_model = gpt_model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.rough_estimate = rough_estimate

    @staticmethod
    def from_openai_usage_dictionary(
        gpt_model: JAImsGPTModel, dictionary: dict
    ) -> JAImsTokensExpense:
        return JAImsTokensExpense(
            gpt_model=gpt_model,
            prompt_tokens=dictionary["prompt_tokens"],
            completion_tokens=dictionary["completion_tokens"],
            total_tokens=dictionary["total_tokens"],
        )

    def spend(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def add_from(self, other_expense: JAImsTokensExpense):
        self.prompt_tokens += other_expense.prompt_tokens
        self.completion_tokens += other_expense.completion_tokens
        self.total_tokens += other_expense.total_tokens
        if other_expense.rough_estimate:
            self.rough_estimate = True  # becomes rough if summed with something rough

    def get_cost(self):
        return (self.prompt_tokens / 1000) * self.gpt_model.price_1k_tokens_in + (
            self.completion_tokens / 1000
        ) * self.gpt_model.price_1k_tokens_out

    def __str__(self):
        string_repr = (
            f"GPT model: {self.gpt_model}\n"
            f"Prompt tokens: {self.prompt_tokens}\n"
            f"Completion tokens: {self.completion_tokens}\n"
            f"Total tokens: {self.total_tokens}\n"
            f"Cost: {round(self.get_cost(),4)}$"
        )

        if self.rough_estimate:
            string_repr += "\n(warning: rough estimate)"

        return string_repr

    def to_json(self):
        return {
            "model": self.gpt_model.string,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.get_cost(),
            "rough_estimate": self.rough_estimate,
        }


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


class JAImsOpenaiKWArgs:
    """
    Represents the keyword arguments for the JAIms OpenAI wrapper.
    This class entirely mirrors the openai API parameters, so refer to it for documentation.
    (https://platform.openai.com/docs/api-reference/chat/create).

    Args:
        model (JAImsGPTModel, optional): The OpenAI model to use. Defaults to JAImsGPTModel.GPT_3_5_TURBO.
        messages (List[dict], optional): The list of messages for the conversation. Defaults to an empty list, it is automatically populated by the run method so it is not necessary to pass them.
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
        tools (Optional[List[JAImsFuncWrapper]], optional): The list of function wrappers to use as tools. Defaults to None.
    """

    def __init__(
        self,
        model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
        messages: List[dict] = [],
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
        tools: Optional[List[JAImsFuncWrapper]] = None,
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
            kwargs["tools"] = parse_function_wrappers_to_tools(self.tools)
            kwargs["tool_choice"] = self.tool_choice

        return kwargs


class JAImsOptions:
    """
    Represents the options for JAImsAgent.

    Args:
        leading_prompts (Optional[List[Dict]]): A list of leading prompts, these will be always prepended to the history for each run.
        trailing_prompts (Optional[List[Dict]]): A list of trailing promptsm, these will be always appended to the history for each run.
        max_consecutive_function_calls (int): The maximum number of consecutive function calls allowed (defaults to 10 to avoid infinite loops).
        optimize_context (bool): Whether to optimize the context in the history manager or not, defaults to True.
        message_history_size (Optional[int]): The size of the message history for each run, only the last n messages will be passed, defaults to none (every message is passed until optimization starts).
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
        leading_prompts: Optional[List[Dict]] = None,
        trailing_prompts: Optional[List[Dict]] = None,
        max_consecutive_function_calls: int = MAX_CONSECUTIVE_CALLS,
        optimize_context: bool = False,
        message_history_size: Optional[int] = None,
        max_retries=15,
        retry_delay=10,
        exponential_base: int = 2,
        exponential_delay: int = 1,
        exponential_cap: Optional[int] = None,
        jitter: bool = True,
        debug_stream_function_call=False,
    ):
        self.leading_prompts = leading_prompts
        self.trailing_prompts = trailing_prompts
        self.max_consecutive_function_calls = max_consecutive_function_calls
        self.optimize_context = optimize_context
        self.message_history_size = message_history_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_base = exponential_base
        self.exponential_delay = exponential_delay
        self.exponential_cap = exponential_cap
        self.jitter = jitter
        self.debug_stream_function_call = debug_stream_function_call


def get_openai_response(
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
