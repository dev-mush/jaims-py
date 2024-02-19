import json
import logging
import os
from typing import Generator, List, Optional, Union

from jaims.transaction_storage import (
    JAImsTransactionStorageInterface,
)

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from jaims.openai_wrappers import (
    estimate_token_count,
    get_openai_response,
)

from jaims.tool_handler import (
    JAImsToolHandler,
)
from jaims.history_manager import HistoryManager

from jaims.entities import (
    JAImsGPTModel,
    JAImsOpenaiKWArgs,
    JAImsOptions,
    JAImsTokensExpense,
    JAImsMissingOpenaiAPIKeyException,
    JAImsMaxConsecutiveFunctionCallsExceeded,
)


class JAImsAgent:
    """
    JAImsAgent realizes the class that interacts with the OpenAI API. It is an agent capable of
    tool calling, with built in history management and token expense tracking.

    Attributes:
        openai_kwargs (JAImsOpenaiKWArgs): The OpenAI keyword arguments.
        options (JAImsOptions): The options for the agent.
        openai_api_key (Optional[str]): The OpenAI API key.
        transaction_storage (JAImsTransactionStorageInterface): The transaction storage interface.

    Methods:
        run: Runs the agent with the given messages and options.
        get_expenses: Returns the tokens spent in the current session (all runs performed on this agent).
        get_last_run_expenses: Returns the tokens spent in the last run of the agent.
        get_run_history: Returns the history that will be sent to OpenAI for a run (returns the messages history passed to openai in the last run).
        get_history: Returns the complete history of the agent.
        clear_history: Clears the history.
    """

    def __init__(
        self,
        openai_kwargs: Optional[JAImsOpenaiKWArgs] = None,
        options: Optional[JAImsOptions] = None,
        openai_api_key: Optional[str] = None,
        tool_handler: Optional[JAImsToolHandler] = None,
        transaction_storage: Optional[JAImsTransactionStorageInterface] = None,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JAImsMissingOpenaiAPIKeyException()
        openai.api_key = openai_api_key

        self.__openai_kwargs = openai_kwargs or JAImsOpenaiKWArgs()
        self.__options = options or JAImsOptions()
        self.__expense = JAImsAgent.__init_expense_dictionary()
        self.__last_run_expense = JAImsAgent.__init_expense_dictionary()
        self.__function_handler = tool_handler or JAImsToolHandler()
        self.__history_manager = HistoryManager()

        if self.__openai_kwargs.messages:
            self.__history_manager.add_messages(self.__openai_kwargs.messages)

        self.__iteration_count = 0
        if transaction_storage is None:
            self.__transaction_storage = JAImsTransactionStorageInterface()
        else:
            self.__transaction_storage = transaction_storage

    def __increment_iteration_count(self):
        self.__iteration_count += 1
        if self.__iteration_count > self.__options.max_consecutive_function_calls:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(
                f"Max consecutive function calls exceeded ({self.__options.max_consecutive_function_calls})"
            )

    def __reset_iteration_count(self):
        self.__iteration_count = -1

    def set_openai_kwargs(self, openai_kwargs: JAImsOpenaiKWArgs):
        """
        Updates the openai kwargs of the agent.

        Args:
            openai_kwargs (JAImsOpenaiKWArgs): The new openai kwargs.
        """
        self.__openai_kwargs = openai_kwargs

    def set_options(self, options: JAImsOptions):
        """
        Updates the options of the agent.

        Args:
            options (JAImsOptions): The new options.
        """
        self.__options = options

    # TODO: add support for multiple responses when n>1
    def run(
        self,
        messages: Optional[List[dict]] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Starts a run calling openai with the passed messages.
        During a run, unless cleared, the history of the previous runs is preserved and optimized (see HistoryManager), unless clear_history is called.

        Args:
            messages (Optional[List[dict]], optional): The messages to be sent to OpenAI.
        """
        self.__reset_iteration_count()
        messages = messages or []
        self.__history_manager.add_messages(messages)

        self.__last_run_expense = JAImsAgent.__init_expense_dictionary()
        return self.__call_openai()

    def __call_openai(self) -> Union[str, Generator[str, None, None]]:
        # throws if max consecutive calls is exceeded
        self.__increment_iteration_count()
        messages = self.__history_manager.get_messages_for_current_run(
            options=self.__options,
            openai_kwargs=self.__openai_kwargs,
        )

        self.__openai_kwargs.messages = messages

        response = get_openai_response(self.__openai_kwargs, self.__options)

        if isinstance(response, openai.Stream):
            return self.__handle_streaming_response(response)
        else:
            return self.__handle_response(response)

    def __handle_streaming_response(
        self,
        response: openai.Stream[ChatCompletionChunk],
    ) -> Generator[str, None, None]:
        accumulated_chunks = None
        for completion_chunk in response:
            if len(completion_chunk.choices) > 0:
                # check content exists
                message_delta = completion_chunk.choices[0].delta
                accumulated_chunks = JAImsAgent.__accumulate_choice_delta(
                    accumulated_chunks, message_delta
                )

                if (
                    self.__options.debug_stream_function_call
                    and message_delta.tool_calls
                ):
                    print(
                        message_delta.tool_calls[0].function.arguments,  # type: ignore
                        flush=True,
                        end="",
                    )

                if completion_chunk.choices[0].finish_reason is not None:
                    if (
                        self.__options.debug_stream_function_call
                        and accumulated_chunks.tool_calls
                    ):
                        print("\n")

                    # rebuilding entire completion chunk with accumulated delta
                    completion_chunk.choices[0].delta = accumulated_chunks
                    self.__store_transaction(completion_chunk)

                    self.__handle_token_expense_from_openai_response(
                        accumulated_chunks.model_dump()
                    )

                    yield from self.__evaluate_next_run_from_message(accumulated_chunks)

                if message_delta.content:
                    yield message_delta.content

    def __handle_response(self, response: ChatCompletion):
        if len(response.choices) == 0:
            return ""

        # log token expense
        self.__handle_token_expense_from_openai_response(response.model_dump())

        self.__store_transaction(response)

        message = response.choices[0].message
        return self.__evaluate_next_run_from_message(message)

    def __evaluate_next_run_from_message(self, message):
        logger = logging.getLogger(__name__)
        logger.debug(f"OpenAI response message:\n{message}")

        message_dict = message.model_dump(exclude_none=True)
        self.__history_manager.add_messages([message_dict])

        if message.tool_calls:
            tools_results = self.__function_handler.handle_tool_calls(
                tool_calls=message_dict.get("tool_calls", []),
                function_wrappers=self.__openai_kwargs.tools or [],
                current_kwargs=self.__openai_kwargs,
                current_options=self.__options,
            )

            if tools_results.override_kwargs:
                self.set_openai_kwargs(tools_results.override_kwargs)

            if tools_results.override_options:
                self.set_options(tools_results.override_options)

            if (
                tools_results.stop == False
                and self.__openai_kwargs.tool_choice == "auto"
            ):
                messages = tools_results.function_result_messages
                messages.extend(
                    tools_results.override_kwargs.messages
                    if tools_results.override_kwargs is not None
                    else []
                )
                self.__history_manager.add_messages(messages)
                return self.__call_openai()

        # if the response was streaming only an empty string must be returned
        # because the streaming generator already yielded the response, otherwise
        # the content is returned.
        if self.__openai_kwargs.stream:
            return ""
        else:
            return message.content or ""

    def __store_transaction(
        self,
        response: Union[ChatCompletion, ChatCompletionChunk],
    ):
        self.__transaction_storage.store_transaction(
            request=self.__openai_kwargs.to_dict(),
            response=response.model_dump(exclude_none=True),
        )

    def __handle_token_expense_from_openai_response(self, response):
        if not self.__openai_kwargs.stream:
            expense = JAImsTokensExpense.from_openai_usage_dictionary(
                self.__openai_kwargs.model, response["usage"]
            )
        else:
            sent_messages = self.__history_manager.get_messages_for_current_run(
                options=self.__options,
                openai_kwargs=self.__openai_kwargs,
            )
            prompt_tokens = estimate_token_count(
                json.dumps(sent_messages), model=self.__openai_kwargs.model
            )
            completion_tokens = estimate_token_count(
                json.dumps(response), model=self.__openai_kwargs.model
            )
            total_tokens = prompt_tokens + completion_tokens
            expense = JAImsTokensExpense(
                gpt_model=self.__openai_kwargs.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                rough_estimate=True,
            )

        self.__last_run_expense[expense.gpt_model.string].add_from(expense)
        self.__expense[expense.gpt_model.string].add_from(expense)

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

    def get_run_history(
        self,
        override_options: Optional[JAImsOptions] = None,
        override_openai_kwargs: Optional[JAImsOpenaiKWArgs] = None,
    ):
        """
        Returns the history that will be sent given the current options and openai kwargs to openai for a run.
        """
        options = override_options or self.__options
        openai_kwargs = override_openai_kwargs or self.__openai_kwargs

        messages = self.__history_manager.get_messages_for_current_run(
            options, openai_kwargs
        )
        return messages

    def get_history(self):
        """
        Returns the complete history of the agent.
        """

        return self.__history_manager.get_history()

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history_manager.clear_history()

    @staticmethod
    def __init_expense_dictionary():
        dict = {}
        for gpt in JAImsGPTModel:
            dict[gpt.string] = JAImsTokensExpense(gpt_model=gpt)

        return dict

    @staticmethod
    def __merge_tool_calls(existing_tool_calls, new_tool_calls_delta):
        if not existing_tool_calls:
            return new_tool_calls_delta

        new_tool_calls = existing_tool_calls[:]
        for new_call_delta in new_tool_calls_delta:
            existing_call = next(
                (item for item in new_tool_calls if item.index == new_call_delta.index),
                None,
            )
            if not existing_call:
                new_tool_calls.append(new_call_delta)
            else:
                if (
                    existing_call.type != new_call_delta.type
                    and new_call_delta.type is not None
                ):
                    existing_call.type = new_call_delta.type
                if (
                    existing_call.id != new_call_delta.id
                    and new_call_delta.id is not None
                ):
                    existing_call.id = new_call_delta.id
                if existing_call.function is None:
                    existing_call.function = new_call_delta.function
                else:
                    if (
                        existing_call.function.name != new_call_delta.function.name
                        and new_call_delta.function.name is not None
                    ):
                        existing_call.function.name = new_call_delta.function.name
                    existing_call.function.arguments = (
                        existing_call.function.arguments or ""
                    ) + (new_call_delta.function.arguments or "")

        return new_tool_calls

    @staticmethod
    def __accumulate_choice_delta(accumulator, new_delta):
        if accumulator is None:
            return new_delta

        if new_delta.content:
            accumulator.content = (accumulator.content or "") + new_delta.content
        if new_delta.role:
            accumulator.role = (accumulator.role or "") + new_delta.role
        if new_delta.tool_calls:
            accumulator.tool_calls = JAImsAgent.__merge_tool_calls(
                accumulator.tool_calls, new_delta.tool_calls
            )

        return accumulator
