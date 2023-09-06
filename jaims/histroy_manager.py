from typing import List, Optional
from jaims.openai_wrappers import (
    DEFAULT_MAX_TOKENS,
    JAImsGPTModel,
    estimate_token_count,
)

from jaims.exceptions import JAImsTokensLimitExceeded
import json

from jaims.function_handler import JAImsFuncWrapper, parse_functions_to_json


class HistoryManager:
    """
    Handles chat history of the agent.

    Attributes
    ----------
        model : GPTModel
            the model to be used by the agent, defaults to gpt-3.5-turbo-0613
        functions : list (optional)
            the list of functions that can be called by the agent, used to compute token usage
        initial_prompts: list (optional)
            the list of initial prompts to be used by the agent, useful to compute token usage
        last_n_turns: int (optional)
            if set, specifies the n last messages to be sent, defaults to None

    Methods
    -------
        add_messages(messages)
            pushes a new message in the history
        get_history() -> list
            returns the history
        clear_history()
            clears the history
        optimize_history()
            optimizes history messages based on context constraints

    Private Members
    ----------------
        __history : list
            holds the current openai messages history. It's meant to be
            manipulated only by the methods of this class.
    """

    def __init__(
        self,
        history: Optional[List] = None,
        model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
        mandatory_context: Optional[List] = None,
        functions: Optional[List[JAImsFuncWrapper]] = None,
        optimize_history: bool = True,
        last_n_turns: Optional[int] = None,
    ):
        self.__history = history or []
        self.model = model
        self.mandatory_context = mandatory_context or []
        funcs_to_parse = functions or []
        self.json_functions = parse_functions_to_json(funcs_to_parse)
        self.optimize_history = optimize_history
        self.last_n_turns = last_n_turns

    def add_messages(self, messages: List):
        """
        Pushes new messages in the history.

        Parameters
        ----------
            message : str
                the message to be added
        """

        if not all(isinstance(message, dict) for message in messages):
            raise TypeError(
                "All messages must be dicts, conforming to OpenAI API specification."
            )

        keys = {"role", "content", "name"}
        parsed = [
            {k: v for k, v in message.items() if k in keys} for message in messages
        ]

        for message, parsed_message in zip(messages, parsed):
            if "function_call" in message:
                function_call = message["function_call"]
                parsed_message["function_call"] = {
                    "name": function_call["name"],
                    "arguments": function_call["arguments"],
                }

        self.__history.extend(parsed)

    def get_optimised_messages(
        self,
        agent_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> List:
        """
        Returns the history.

        Parameters
        ----------
            agent_max_tokens : int (optional)
                the max tokens to leave out for the response from openai, defaults to DEFAULT_MAX_TOKENS

        Returns
        -------
            list
                the history of messages to be sent to openai

        Raises
        ------
            TokensLimitExceeded
                if the max tokens to be used exceed the max tokens supported by the current llm model

        Developer Notes
        ---------------

        # Optimization Feature
        This is roughly how the optimization feature works right now, plus some notes on how I intend to improve it.
        The tokens for the messages to send to openai are calculated by composing the mandatory_context, the chat history managed by this class
        and the functions in a variable named compound_history.
        They are parsed to a json string and tiktoken evaluates the token consumption based on the current llm model setting.

        Right now I'm parsing the whole compound_history, including the functions, to json and calculating the tokens based on the json string,
        but I'm not sure how accurate this is.
        On rough estimates made with the tokenizer from openai they seem to be pretty accurate, but I need to investigate more.

        A high level overview of the optimization loop is the following:

        the max_tokens to be used are the max tokens supported by the current llm model, minus the tokens to leave out for the response
        from openai passed in agent_max_tokens (passed with a default value) .

        The compound_history tokens are calculated, and while they exceed the context max_tokens:
        1. the first (oldest) message from the chat history between the user and the angent is popped
        2. the compound_history tokens are recalculated
        3. if the compound_history tokens are still above the context max_tokens, the process is repeated from step 1
            3.1. if the chat history between user and agent remains empty for some reason
                (this could happen for instance if functions and mandatory context are way too big),
                an exception is raised. I think it aids development but have to think about it.
        4. if the compound_history tokens are below the context max_tokens, the mandatory_context + the optimized history are returned.

        TODO:
        - optimize functions messages in the history

        MAYBE TODO:
        - it would be nice to have an auto-scale up of the context, for instance passing from the gpt-3.5-turbo 4k to the 16k model.



        """

        # Copying the whole history to avoid altering the original one
        history_buffer = self.__history.copy()

        # If last_n_turns is set, only keep the last n messages
        if self.last_n_turns is not None:
            history_buffer = history_buffer[-self.last_n_turns :]

        # create the compound history with the mandatory context
        # the actual chat history and the functions to calculate the tokens
        compound_history = (
            self.mandatory_context + history_buffer + (self.json_functions)
        )

        # the max tokens to be used are the max tokens supported by the current
        # openai model minus the tokens to leave out for the response from openai
        context_max_tokens = self.model.max_tokens - agent_max_tokens

        # calculate the tokens for the compound history
        messages_tokens = self.__tokens_from_messages(compound_history)

        if self.optimize_history:
            while messages_tokens > context_max_tokens:
                if not history_buffer:
                    raise JAImsTokensLimitExceeded(
                        self.model.max_tokens,
                        messages_tokens,
                        agent_max_tokens,
                        has_optimized=True,
                    )

                # Popping the first (oldest) message from the chat history between the user and agent
                history_buffer.pop(0)

                # Recalculating the tokens for the compound history
                messages_tokens = self.__tokens_from_messages(
                    self.mandatory_context + history_buffer + self.json_functions
                )
        elif messages_tokens > context_max_tokens:
            raise JAImsTokensLimitExceeded(
                self.model.max_tokens,
                messages_tokens,
                agent_max_tokens,
                has_optimized=False,
            )

        llm_messages = self.mandatory_context + history_buffer

        return llm_messages

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history = []

    def get_history(self, optimized=False):
        if optimized:
            return self.get_optimised_messages()

        return self.__history

    def __tokens_from_messages(self, messages: List):
        """Returns the number of tokens used by a list of messages."""
        return estimate_token_count(json.dumps(messages), self.model)
