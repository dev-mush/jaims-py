import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Union

import openai

from jaims.openai_wrappers import (
    DEFAULT_MAX_TOKENS,
    MAX_CONSECUTIVE_CALLS,
    JAImsGPTModel,
    JAImsTokensExpense,
    estimate_token_count,
    get_openai_response,
)
from jaims.exceptions import (
    JAImsMissingOpenaiAPIKeyException,
    JAImsMaxConsecutiveFunctionCallsExceeded,
)
from jaims.function_handler import (
    JAImsFuncWrapper,
    JAImsFunctionHandler,
    parse_functions_to_json,
)
from jaims.histroy_manager import HistoryManager


# private class used to store a call context when looping happens because of function
class OpenaiCallContext:
    def __init__(self, call_kwargs: dict, iterations: int = 0):
        self.call_kwargs = call_kwargs
        self.iterations = iterations


class JAImsAgent:
    """
    A simple agent, gets initialized with the model and a list of function that
    can be called, and returns the response. Handles the conversation history by itself.

    Attributes
    ----------
        model : GPTModel
            the model to be used by the agent, defaults to gpt-3.5-turbo-0613
        functions : list
            the list of functions that can be called by the agent
        initial_prompts: list (optional)
            the list of initial prompts to be used by the agent, useful to inject
            some system messages that shape the personality or the scope of the agent
        optimize_context: bool
            wether to optimize the context for each call to OpenAi or not. It is useful when the agent
            is used for instance as a conversational bot, this parameter ensures that, when reaching context limits,
            older messages will be popped from the history before being passed to OpenAI, defaults to True.
            When set to false, once the context limit is reached, an exception will be raised.
        last_n_turns: int
            The number of last messages that has to be kept in the history, default (0) is uncapped.
            Uncapped means that the history is sent in full as long as it fits the context and then, depending
            on the value of optimize_context, it will be optimized trimming the oldest messages or an exception
            will be raised when saturated.
        max_consecutive_calls: int
            the maximum number of consecutive function calls that can be made by the agent, defaults to 5
            for safety reasons, to avoid unwanted loops that might impact up token usage
        openai_api_key: str
            the openai api key, defaults to the OPENAI_API_KEY environment variable if not provided

    Methods
    -------
        run(messages, stream: bool optional) -> JAImsResponse
            performs the call to OpenAI and returns the response
        clear_history()
            clears the agent history
        get_history(optimized: bool) -> List[dict]
            returns the current history of the agent, if optimized is passed to True, it will return the
            optimized version, otherwise the full history since the beginning of this agent session
        get_expenses() -> List[JaimsTokensExpense]
            returns the currently spent tokens for this agent session, one for each model used


    Raises
    ------
        MissingOpenaiAPIKeyException
            if the OPENAI_API_KEY environment variable is not set and no api key is provided

    Private Members
    ---------------
        __history_manager : HistoryManager
            the history manager
    """

    def __init__(
        self,
        model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
        functions: Optional[List[JAImsFuncWrapper]] = None,
        initial_prompts: Optional[List[Dict]] = None,
        max_consecutive_calls: int = MAX_CONSECUTIVE_CALLS,
        optimize_context: bool = True,
        last_n_turns: Optional[int] = None,
        openai_api_key: Optional[str] = None,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JAImsMissingOpenaiAPIKeyException()
        openai.api_key = openai_api_key

        self.model = model
        self.functions = functions or []
        self.initial_prompts = initial_prompts or []
        self.max_consecutive_calls = max_consecutive_calls
        self.__expense = {
            JAImsGPTModel.GPT_3_5_TURBO.string: JAImsTokensExpense(
                gpt_model=JAImsGPTModel.GPT_3_5_TURBO
            ),
            JAImsGPTModel.GPT_3_5_TURBO_16K.string: JAImsTokensExpense(
                gpt_model=JAImsGPTModel.GPT_3_5_TURBO_16K
            ),
            JAImsGPTModel.GPT_4.string: JAImsTokensExpense(
                gpt_model=JAImsGPTModel.GPT_4
            ),
        }
        self.__function_handler = JAImsFunctionHandler(self.functions)
        self.__history_manager = HistoryManager(
            model=self.model,
            functions=self.functions,
            mandatory_context=self.initial_prompts,
            optimize_history=optimize_context,
            last_n_turns=last_n_turns,
        )

    def run(
        self,
        messages: Optional[List[dict]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        temperature: float = 0.0,
        top_p: Optional[int] = None,
        n: int = 1,
        function_call: Union[str, Dict] = "auto",
        max_retries: int = 15,
        delay: int = 10,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Calls OpenAI with the passed parameters and returns or streams the response.

        Parameters
        ----------
            messages : list
                the list of messages to be sent
            stream : bool (optional)
                whether to stream the response or not, defaults to False
            max_tokens : int (optional)
                the maximum tokens to be used to generate the answer
                defaults to 512
            temperature : float (optional)
                the temperature to be used to generate the answer
                defaults to 0.0
            top_p : float (optional)
                the top_p to be used to generate the answer
                defaults to None
            n : int (optional)
                the number of answers to be generated
                defaults to 1
            function_call : str (optional)
                the function call to be used, defaults to "auto"
            max_retries : int
                the maximum number of retries to be used when calling OpenAI in case of error
                defaults to 15
            delay : int
                the delay in seconds to be used between retries
                defaults to 10

        Returns
        -------
            JAImsResponse
                the response object
        """

        self.__history_manager.add_messages(messages or [])

        kwargs = {
            "model": self.model,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
            "n": n,
            "max_retries": max_retries,
            "delay": delay,
        }

        if self.functions:
            kwargs["function_call"] = function_call
            kwargs["functions"] = parse_functions_to_json(self.functions)

        call_context = OpenaiCallContext(kwargs)

        return self.__call_openai(call_context)

    def __call_openai(
        self, call_context: OpenaiCallContext
    ) -> Union[str, Generator[str, None, None]]:
        call_context.iterations += 1
        if call_context.iterations > self.max_consecutive_calls:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(
                f"Max consecutive function calls exceeded ({self.max_consecutive_calls})"
            )

        messages = self.__history_manager.get_optimised_messages(
            agent_max_tokens=call_context.call_kwargs["max_tokens"],
        )

        call_context.call_kwargs["messages"] = messages

        response = get_openai_response(**call_context.call_kwargs)

        if call_context.call_kwargs["stream"]:
            return self.__process_openai_stream_response(response, call_context)
        else:
            return self.__process_openai_response(response, call_context)

    def __process_openai_stream_response(
        self, response: Any, call_context: OpenaiCallContext
    ) -> Generator[str, None, None]:
        message = {}
        for response_delta in response:
            if len(response_delta["choices"]) > 0:
                # check content exists
                message_delta = response_delta["choices"][0]["delta"]
                message = JAImsAgent.__merge_message_deltas(message, message_delta)

                if response_delta["choices"][0]["finish_reason"] is not None:
                    # log token expense
                    sent_messages = self.__history_manager.get_optimised_messages(
                        agent_max_tokens=call_context.call_kwargs["max_tokens"],
                    )
                    prompt_tokens = estimate_token_count(
                        json.dumps(sent_messages), model=self.model
                    )
                    completion_tokens = estimate_token_count(
                        json.dumps(message), model=self.model
                    )
                    total_tokens = prompt_tokens + completion_tokens
                    rough_expense = JAImsTokensExpense(
                        gpt_model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        rough_estimate=True,
                    )
                    self.__log_new_expense(rough_expense)

                    yield from self.__handle_response_message(message, call_context)

                if "content" in message_delta and message_delta["content"] is not None:
                    yield response_delta["choices"][0]["delta"]["content"]

    def __process_openai_response(self, response: Any, call_context: OpenaiCallContext):
        if len(response["choices"]) == 0:
            return ""

        # log token expense
        expense = JAImsTokensExpense.from_openai_usage_dictionary(
            self.model, response["usage"]
        )
        self.__log_new_expense(expense)

        message = response["choices"][0]["message"]
        return self.__handle_response_message(message, call_context)

    def __handle_response_message(self, message, call_context):
        self.__history_manager.add_messages([message])
        logger = logging.getLogger(__name__)
        logger.debug(f"OpenAI response:\n{message}")

        if "function_call" in message:
            result_message = self.__function_handler.handle_from_message(
                message=message
            )
            if call_context.call_kwargs["function_call"] == "auto":
                self.__history_manager.add_messages([result_message])
                return self.__call_openai(call_context)

        if call_context.call_kwargs["stream"]:
            return ""

        return message["content"]

    def __log_new_expense(self, expense: JAImsTokensExpense):
        self.__expense[expense.gpt_model.string].add_from(expense)

    def get_expenses(self):
        """
        Returns the tokens spent in the current session in an array, one for each model.
        """
        return [
            expense for expense in self.__expense.values() if expense.total_tokens > 0
        ]

    def get_history(self, optimized: bool = False):
        """
        Returns the current history of the agent.
        """
        return self.__history_manager.get_history(optimized=optimized)

    @staticmethod
    def __merge_message_deltas(current: dict, delta: dict) -> dict:
        """Merges new delta into the accumulated one."""
        for key, value in delta.items():
            if key in current:
                if isinstance(value, str):
                    current[key] += value
                elif isinstance(value, dict):
                    current[key] = JAImsAgent.__merge_message_deltas(
                        current.get(key, {}), value
                    )
            else:
                current[key] = value

        return current
