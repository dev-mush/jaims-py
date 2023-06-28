from enum import Enum
from typing import List, Optional
from core.constants import DEFAULT_MAX_TOKENS, GPTModel

from core.exceptions import TokensLimitExceeded
import tiktoken
import json


class HistoryManager:
    """
    Handles chat history of the agent.

    Attributes
    ----------
        __history : list
          holds the current openai messages history. It's meant to be
          manipulated only by the methods of this class.
        model : GPTModel
            the model to be used by the agent, defaults to gpt-3.5-turbo-0613

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
    """

    def __init__(self, history: List = [], model: GPTModel = GPTModel.GPT_3_5_TURBO):
        self.__history = history
        self.model = model

    def add_messages(self, messages: List):
        """
        Pushes new messages in the history.

        Parameters
        ----------
            message : str
                the message to be added
        """
        self.__history.extend(messages)

    def build_messages_from_history(
        self,
        mandatory_context: Optional[List] = None,
        functions: Optional[List] = None,
        agent_max_tokens: int = DEFAULT_MAX_TOKENS,
        optimize: bool = False,
    ) -> List:
        """
        Returns the history.

        Parameters
        ----------
            mandatory_context : list (optional)
                The list of mandatory context messages to be prepended to the message history.
                Useful to inject some system messages that shape the personality or the scope of the agent.
            functions : list (optional)
                The list of functions to be appended to the message history, pass it when needed to calculate tokens and optimize the request.
            optimize : bool (optional)
                Whether to optimize the history or not, defaults to False.
                When set to True, the history will be optimized based on context limits for the current model.

        Returns
        -------
            list
                the history

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

        # Assigning empty lists to mandatory_context and functions if they are None
        mandatory_context = mandatory_context or []
        functions = functions or []

        # Copying the whole history to avoid altering the original one
        history_buffer = self.__history.copy()

        # create the compound history with the mandatory context
        # the actual chat history and the functions to calculate the tokens
        compound_history = mandatory_context + history_buffer + functions

        # the max tokens to be used are the max tokens supported by the current
        # openai model minus the tokens to leave out for the response from openai
        context_max_tokens = self.model.max_tokens - agent_max_tokens

        # calculate the tokens for the compound history
        messages_tokens = self.__tokens_from_messages(compound_history)

        if optimize:
            while messages_tokens > context_max_tokens:
                if not history_buffer:
                    raise TokensLimitExceeded(
                        self.model.max_tokens,
                        messages_tokens,
                        agent_max_tokens,
                        has_optimized=True,
                    )

                # Popping the first (oldest) message from the chat history between the user and agent
                history_buffer.pop(0)

                # Recalculating the tokens for the compound history
                messages_tokens = self.__tokens_from_messages(
                    mandatory_context + history_buffer + functions
                )
        elif messages_tokens > context_max_tokens:
            raise TokensLimitExceeded(
                self.model.max_tokens,
                messages_tokens,
                agent_max_tokens,
                has_optimized=False,
            )

        return mandatory_context + history_buffer

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history = []

    def __tokens_from_messages(self, messages: List):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model.string)
        except KeyError:
            print("Warning: model not found. Using gpt-3.5-turbo encoding.")
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")

        messages_json = json.dumps(messages)
        messages_tokens = len(encoding.encode(messages_json))
        return messages_tokens
