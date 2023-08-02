from __future__ import annotations

from enum import Enum
import time
from typing import Any, List, Optional, Dict, Union
import openai
import tiktoken
import logging

DEFAULT_MAX_TOKENS = 512
MAX_CONSECUTIVE_CALLS = 5


class JAImsGPTModel(Enum):
    """
    The OPENAI GPT models available.
    Only those that support functions are listed, so just:
    gpt-3.5-turbo-0613, gpt-3-5-turbo-16k-0613, gpt-4-0613
    """

    GPT_3_5_TURBO = ("gpt-3.5-turbo-0613", 4096, 0.0015, 0.002)
    GPT_3_5_TURBO_16K = ("gpt-3.5-turbo-16k-0613", 16384, 0.003, 0.004)
    GPT_4 = ("gpt-4-0613", 8192, 0.03, 0.06)

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


def get_openai_response(
    messages,
    model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    temperature: float = 0.0,
    top_p: Optional[int] = None,
    n: int = 1,
    function_call: Union[str, Dict] = "auto",
    functions: Optional[List[Dict[str, Any]]] = None,
    max_retries=15,
    delay=10,
):
    retries = 0
    logger = logging.getLogger(__name__)

    kwargs = {
        "model": model.string,
        "temperature": temperature,
        "n": n,
        "stream": stream,
        "messages": messages,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if functions:
        kwargs["functions"] = functions
        kwargs["function_call"] = function_call
        logger.info("Using functions")
        logger.debug(f"Function call:\n{kwargs['function_call']}")
        logger.debug(kwargs["functions"])

    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                **kwargs,
            )

            return response
        except openai.OpenAIError as error:
            logger.error(f"OpenAI API error code: {error.code}")
            logger.error(f"OpenAI API error:\n{error}\n")

            # check if error contains the string "Rate limit"
            if "rate limit" in str(error).lower():
                logger.warning(f"Retrying in 15 seconds...")
                time.sleep(15)
            else:
                logger.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            retries += 1

    max_retries_error = (
        f"Max retries exceeded! OpenAI API call failed {max_retries} times."
    )
    logger.error(max_retries_error)
    raise Exception(max_retries_error)
