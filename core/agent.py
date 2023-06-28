import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

import openai

from core.constants import DEFAULT_MAX_TOKENS, GPTModel
from core.exceptions import (
    MissingOpenaiAPIKeyException,
    OpenAIErrorException,
    UnexpectedFunctionCall,
)
from core.func_wrapper import JAImsFuncWrapper
from core.histroy_manager import HistoryManager

# TODO: Parametrize temperature and top_p
# TODO: Implement function calling
#  - Create dummy function to inject with function wrapper
#  - pass the dummy function
#  - implement function loop:
#    - if streaming get the function call from the response, call the function and pass the message back until a normal response is received
#    - if not streaming accumulate the function call from the response, call the function and pass the message back until a normal response is received
#


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

    Methods
    -------
        send_messages(messages, stream: bool optional) -> JAImsResponse
            sends a list of messages to GPT and returns the response
        clear_history()
            clears the agent history

    Private Members
    ---------------
        __history_manager : HistoryManager
            the history manager
    """

    # private class used to store a call context when looping happens because of function
    class __OpenaiCallContext:
        def __init__(self, stream: bool, max_tokens: int):
            self.stream = stream
            self.max_tokens = max_tokens

    def __init__(
        self,
        model=GPTModel.GPT_3_5_TURBO,
        functions: List[JAImsFuncWrapper] = [],
        initial_prompts: Optional[List[Dict]] = [],
        openai_api_key: Optional[str] = None,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise MissingOpenaiAPIKeyException()
        openai.api_key = openai_api_key

        self.model = model
        self.functions = functions
        self.initial_prompts = initial_prompts
        self.__history_manager = HistoryManager(model=self.model)

    @staticmethod
    def __build_functions(functions):
        if not functions:
            return None

        openai_functions = []
        for function in functions:
            function_data = {
                k: v
                for k, v in {
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.get_jsonapi_schema(),
                }.items()
                if v is not None
            }

            openai_functions.append(function_data)

        return openai_functions

    def send_messages(
        self, messages, max_tokens=DEFAULT_MAX_TOKENS, stream=False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Sends a list of messages to GPT and returns the response.

        Parameters
        ----------
            messages : list
                the list of messages to be sent
            stream : bool (optional)
                whether to stream the response or not, defaults to False
            response_buffer : int (optional)
                how much spase to leave in the context for the Agent Response when sending a new message.
                defaults to 512

        Returns
        -------
            JAImsResponse
                the response object
        """
        parsed_functions = JAImsAgent.__build_functions(self.functions)
        self.__history_manager.add_messages(messages)

        optimized_messages = self.__history_manager.build_messages_from_history(
            mandatory_context=self.initial_prompts,
            functions=parsed_functions,
            agent_max_tokens=max_tokens,
            optimize=True,
        )

        kwargs = {
            "model": self.model.string,
            "messages": optimized_messages,
            "temperature": 0.0,
            "stream": stream,
            "max_tokens": max_tokens,
            "n": 1,
        }

        if parsed_functions:
            kwargs["function_call"] = "auto"
            kwargs["functions"] = parsed_functions

        try:
            response: Any = openai.ChatCompletion.create(**kwargs)
            call_context = self.__OpenaiCallContext(stream, max_tokens)

            if stream:
                return self.__answer_with_stream(response, call_context)
            else:
                return self.__answer_no_stream(response, call_context)
        except openai.OpenAIError as e:
            raise OpenAIErrorException(
                f"Failed to communicate with the OpenAI API: {str(e)}", e
            ) from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}") from e

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

    def __answer_with_stream(
        self, response: Any, call_context: __OpenaiCallContext
    ) -> Generator[str, None, None]:
        message = {}
        for response_delta in response:
            if len(response_delta["choices"]) > 0:
                # check content exists
                message_delta = response_delta["choices"][0]["delta"]
                message = JAImsAgent.__merge_message_deltas(message, message_delta)

                if response_delta["choices"][0]["finish_reason"] is not None:
                    self.__history_manager.add_messages([message])
                    if "function_call" in message:
                        yield from self.__handle_function_call(message, call_context)

                if "content" in message_delta and message_delta["content"] is not None:
                    yield response_delta["choices"][0]["delta"]["content"]

    def __handle_function_call(self, message: dict, call_context: __OpenaiCallContext):
        function_name = message["function_call"]["name"]
        function_args = message["function_call"]["arguments"]

        dict_args = json.loads(function_args)

        # invoke function
        call_result = self.call_function_by_name(function_name, **dict_args)

        # build function result message, call new send recursively
        function_result_message = {
            "content": str(call_result),
            "name": function_name,
            "role": "function",
        }

        return self.send_messages(
            [function_result_message],
            stream=call_context.stream,
            max_tokens=call_context.max_tokens,
        )

    def __answer_no_stream(self, response: Any, call_context: __OpenaiCallContext):
        if len(response["choices"]) == 0:
            return ""
        message = response["choices"][0]["message"]
        self.__history_manager.add_messages([message])

        # evaluate method contains function call
        if "function_call" in message:
            return self.__handle_function_call(message, call_context)

        return message["content"]

    def call_function_by_name(self, function_name, *args, **kwargs):
        # Check if function_name exists in functions, if not, raise UnexpectedFunctionCallException
        function_wrapper = next(
            (f for f in self.functions if f.name == function_name), None
        )
        if not function_wrapper:
            raise UnexpectedFunctionCall(function_name)

        # If the name of the current function matches the provided name
        # Call the function and return its result
        return function_wrapper.function(*args, **kwargs)
