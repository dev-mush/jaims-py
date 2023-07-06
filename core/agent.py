import os
from typing import Any, Dict, Generator, List, Optional, Union

import openai

from core.constants import DEFAULT_MAX_TOKENS, MAX_CONSECUTIVE_CALLS, JAImsGPTModel
from core.exceptions import (
    JAImsMissingOpenaiAPIKeyException,
    JAImsOpenAIErrorException,
    JAImsMaxConsecutiveFunctionCallsExceeded,
)
from core.function_handler import (
    JAImsFuncWrapper,
    JAImsFunctionHandler,
    parse_functions_to_json,
)
from core.histroy_manager import HistoryManager


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
        max_consecutive_calls: int
            the maximum number of consecutive function calls that can be made by the agent, defaults to 5
            for safety reasons, to avoid unwanted loops that might impact up token usage
        openai_api_key: str
            the openai api key, defaults to the OPENAI_API_KEY environment variable if not provided

    Methods
    -------
        send_messages(messages, stream: bool optional) -> JAImsResponse
            sends a list of messages to GPT and returns the response
        clear_history()
            clears the agent history

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
        model=JAImsGPTModel.GPT_3_5_TURBO,
        functions: List[JAImsFuncWrapper] = [],
        initial_prompts: Optional[List[Dict]] = [],
        openai_api_key: Optional[str] = None,
        max_consecutive_calls=MAX_CONSECUTIVE_CALLS,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise JAImsMissingOpenaiAPIKeyException()
        openai.api_key = openai_api_key

        self.model = model
        self.functions = functions
        self.initial_prompts = initial_prompts
        self.max_consecutive_calls = max_consecutive_calls
        self.__function_handler = JAImsFunctionHandler(self.functions)
        self.__history_manager = HistoryManager(
            model=self.model,
            functions=self.functions,
            mandatory_context=self.initial_prompts,
        )

    def send_messages(
        self,
        messages,
        max_tokens=DEFAULT_MAX_TOKENS,
        stream=False,
        temperature=0.0,
        top_p=None,
        n=1,
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

        self.__history_manager.add_messages(messages)

        kwargs = {
            "model": self.model.string,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
            "n": n,
        }

        if self.functions:
            kwargs["function_call"] = "auto"
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

        optimized_messages = self.__history_manager.build_messages_from_history(
            agent_max_tokens=call_context.call_kwargs["max_tokens"],
        )

        call_context.call_kwargs["messages"] = optimized_messages

        try:
            response: Any = openai.ChatCompletion.create(**call_context.call_kwargs)

            if call_context.call_kwargs["stream"]:
                return self.__process_openai_stream_response(response, call_context)
            else:
                return self.__process_openai_response(response, call_context)
        except openai.OpenAIError as e:
            raise JAImsOpenAIErrorException(
                f"Failed to communicate with the OpenAI API: {str(e)}", e
            ) from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}") from e

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
                    yield from self.__handle_response(message, call_context)

                if "content" in message_delta and message_delta["content"] is not None:
                    yield response_delta["choices"][0]["delta"]["content"]

    def __process_openai_response(self, response: Any, call_context: OpenaiCallContext):
        if len(response["choices"]) == 0:
            return ""
        message = response["choices"][0]["message"]
        return self.__handle_response(message, call_context)

    def __handle_response(self, message, call_context):
        self.__history_manager.add_messages([message])

        if "function_call" in message:
            result_message = self.__function_handler.handle_from_message(
                message=message
            )
            self.__history_manager.add_messages([result_message])
            return self.__call_openai(call_context)

        if call_context.call_kwargs["stream"]:
            return ""

        return message["content"]

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
