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

from jaims.entities import JAImsOpenaiKWArgs, JAImsOptions, JAImsGPTModel


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
