import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Union

import openai

from jaims.openai_wrappers import (
    JAImsGPTModel,
    JAImsTokensExpense,
    JAImsOpenaiKWArgs,
    JAImsOptions,
    estimate_token_count,
    get_openai_response,
)
from jaims.exceptions import (
    JAImsMissingOpenaiAPIKeyException,
    JAImsMaxConsecutiveFunctionCallsExceeded,
)
from jaims.function_handler import (
    JAImsFunctionHandler,
)
from jaims.histroy_manager import HistoryManager


# private class used to store a call context when looping happens because of function
class JAImsCallContext:
    def __init__(
        self,
        openai_kwargs: JAImsOpenaiKWArgs,
        call_options: JAImsOptions,
        iterations: int = 0,
    ):
        self.openai_kwargs = openai_kwargs
        self.call_options = call_options
        self.iterations = iterations

    def add_iteration(self):
        self.iterations += 1
        if self.iterations > self.call_options.max_consecutive_function_calls:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(
                f"Max consecutive function calls exceeded ({self.call_options.max_consecutive_function_calls})"
            )


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
        get_last_run_expenses() -> List[JaimsTokensExpense]
            returns the spent tokens for the last run of the agent, one for each model used, it is reset at each run
        get_openai_responses() -> List[dict]
            returns the list of raw responses from OpenAI for the entire session
        get_openai_last_run_responses() -> List[dict]
            returns the list of raw responses from OpenAI for the last run of the agent, it is reset at each run


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
        openai_kwargs: JAImsOpenaiKWArgs = JAImsOpenaiKWArgs(),
        options: JAImsOptions = JAImsOptions(),
        openai_api_key: Optional[str] = None,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JAImsMissingOpenaiAPIKeyException()
        openai.api_key = openai_api_key

        self.__openai_kwargs = openai_kwargs
        self.__options = options
        self.__expense = JAImsAgent.__init_expense_dictionary()
        self.__last_run_expense = JAImsAgent.__init_expense_dictionary()
        self.__function_handler = JAImsFunctionHandler()
        self.__history_manager = HistoryManager()
        self.__openai_last_run_responses = []
        self.__openai_responses = []

    def run(
        self,
        messages: Optional[List[dict]] = None,
        override_options: Optional[JAImsOptions] = None,
        override_openai_kwargs: Optional[JAImsOpenaiKWArgs] = None,
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

        messages = messages or []
        if override_openai_kwargs:
            messages = override_openai_kwargs.messages

        self.__history_manager.add_messages(messages)
        options = override_options or self.__options
        openai_kwargs = override_openai_kwargs or self.__openai_kwargs
        call_context = JAImsCallContext(openai_kwargs, options)

        self.__openai_last_run_responses = []
        self.__last_run_expense = JAImsAgent.__init_expense_dictionary()
        return self.__call_openai(call_context)

    def __call_openai(
        self, call_context: JAImsCallContext
    ) -> Union[str, Generator[str, None, None]]:
        # throws if max consecutive calls is exceeded
        call_context.add_iteration()
        messages = self.__history_manager.get_messages_for_current_run(
            options=call_context.call_options,
            openai_kwargs=call_context.openai_kwargs,
        )

        call_context.openai_kwargs.messages = messages

        response = get_openai_response(
            call_context.openai_kwargs, call_context.call_options
        )
        self.__openai_last_run_responses.append(response)
        self.__openai_responses.append(response)

        if call_context.openai_kwargs.stream:
            return self.__process_openai_stream_response(response, call_context)
        else:
            return self.__process_openai_response(response, call_context)

    def __process_openai_stream_response(
        self, response: Any, call_context: JAImsCallContext
    ) -> Generator[str, None, None]:
        message = {}
        for response_delta in response:
            if len(response_delta["choices"]) > 0:
                # check content exists
                message_delta = response_delta["choices"][0]["delta"]
                message = JAImsAgent.__merge_message_deltas(message, message_delta)

                if call_context.call_options.debug_stream_function_call:
                    print(
                        message_delta["function_call"]["arguments"], end="", flush=True
                    )

                if response_delta["choices"][0]["finish_reason"] is not None:
                    # log token expense
                    self.__handle_token_expense_from_openai_response(
                        response_delta, call_context
                    )

                    yield from self.__handle_response_message(message, call_context)

                if "content" in message_delta and message_delta["content"] is not None:
                    yield response_delta["choices"][0]["delta"]["content"]

    def __process_openai_response(self, response: Any, call_context: JAImsCallContext):
        if len(response["choices"]) == 0:
            return ""

        # log token expense
        self.__handle_token_expense_from_openai_response(response, call_context)

        message = response["choices"][0]["message"]
        return self.__handle_response_message(message, call_context)

    def __handle_response_message(self, message, call_context: JAImsCallContext):
        self.__history_manager.add_messages([message])
        logger = logging.getLogger(__name__)
        logger.debug(f"OpenAI response:\n{message}")

        if "function_call" in message:
            result_message = self.__function_handler.handle_from_message(
                message=message,
                functions=call_context.openai_kwargs.functions,
            )
            if call_context.openai_kwargs.function_call == "auto":
                self.__history_manager.add_messages([result_message])
                return self.__call_openai(call_context)

        if call_context.openai_kwargs.stream:
            return ""

        return message["content"]

    def __handle_token_expense_from_openai_response(
        self, response, call_context: JAImsCallContext
    ):
        if not call_context.openai_kwargs.stream:
            expense = JAImsTokensExpense.from_openai_usage_dictionary(
                call_context.openai_kwargs.model, response["usage"]
            )
        else:
            sent_messages = self.__history_manager.get_messages_for_current_run(
                options=call_context.call_options,
                openai_kwargs=call_context.openai_kwargs,
            )
            prompt_tokens = estimate_token_count(
                json.dumps(sent_messages), model=call_context.openai_kwargs.model
            )
            completion_tokens = estimate_token_count(
                json.dumps(response), model=call_context.openai_kwargs.model
            )
            total_tokens = prompt_tokens + completion_tokens
            expense = JAImsTokensExpense(
                gpt_model=call_context.openai_kwargs.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                rough_estimate=True,
            )

        self.__last_run_expense[expense.gpt_model.string].add_from(expense)
        self.__expense[expense.gpt_model.string].add_from(expense)

    def get_openai_responses(self) -> List[dict]:
        """
        Returns the list of raw responses from OpenAI.
        """
        return self.__openai_responses

    def get_openai_last_run_responses(self) -> List[dict]:
        """
        Returns the list of raw responses from OpenAI for the last run of the agent.
        """
        return self.__openai_last_run_responses

    def get_expenses(self):
        """
        Returns the tokens spent in the current session in an array, one for each model.
        """
        return [
            expense for expense in self.__expense.values() if expense.total_tokens > 0
        ]

    def get_last_run_expenses(self):
        """
        Returns the tokens spent in the last run of the agent in an array, one for each model.
        """
        return [
            expense
            for expense in self.__last_run_expense.values()
            if expense.total_tokens > 0
        ]

    def get_history(self, optimized: bool = False):
        """
        Returns the current history of the agent.
        """
        return self.__history_manager.get_history(optimized=optimized)

    @staticmethod
    def __init_expense_dictionary():
        return {
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
