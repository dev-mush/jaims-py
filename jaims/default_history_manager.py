from typing import List, Optional

from jaims.entities import Message
from jaims.interfaces import HistoryManagerITF, HistoryOptimizerITF


class LastNHistoryOptimizer(HistoryOptimizerITF):
    """
    A history optimizer that keeps only the last N messages in the history.
    """

    def __init__(self, last_n: int):
        self.__last_n = last_n

    def optimize_history(self, messages: List[Message]) -> List[Message]:
        """
        Optimizes the history by keeping only the last N messages.

        Args:
            messages (List[Message]): The list of messages to optimize.

        Returns:
            List[Message]: The optimized list of messages.
        """
        return messages[-self.__last_n :] if self.__last_n > 0 else []


class DefaultHistoryManager(HistoryManagerITF):
    """
    A class that handles and manages the history of messages when used as a chat-agent.

    Use the attribtutes leading_prompts and trailing_prompts to add prompts that will be added to the beginning and end of the history at every call (e.g.: For setting system instructions or a fixed trailing prompt).

    The method get_messages retrieves the messages in the history that will be sent to the chat agent at a given invocation, optionally optimized by a provided history optimizer.

    Args:
        history (Optional[List[Message]]): The initial history (usually it is empty, unless you want to start with a predefined history).
        leading_prompts (Optional[List[Message]]): The list of prompts to be added to the beginning of the history at every call.
        trailing_prompts (Optional[List[Message]]): The list of prompts to be added to the end of the history at every call.
        history_optimizer (Optional[HistoryOptimizerInterface]): The history optimizer to be used to optimize
    """

    def __init__(
        self,
        history: Optional[List[Message]] = None,
        leading_prompts: Optional[List[Message]] = None,
        trailing_prompts: Optional[List[Message]] = None,
        history_optimizer: Optional[HistoryOptimizerITF] = None,
    ):
        self.__history = history or []
        self.__leading_prompts = leading_prompts or []
        self.__trailing_prompts = trailing_prompts or []
        self.__history_optimizer = history_optimizer

    def add_messages(self, messages: List[Message]):
        """
        Adds a list of messages to the history.

        Args:
            messages (List[Message]): The list of messages to be added to the history.
        """
        self.__history.extend(messages)

    def get_messages(self):
        """
        Returns the list of messages that will be sent to the chat agent at a given invocation (optionally optimized by the history optimizer). Includes the leading and trailing prompts.

        Do not confuse this method with get_history, which returns the entire history from the beginning of the conversation. If an optimizer is set, this method will return the optimized history for the current invocation, not the entire history. The agent instance invokes this method to get the messages to be sent to the chat agent at a given invocation.

        Returns:
            List[Message]: The list of messages to be sent to the chat agent.
        """
        if self.__history_optimizer:
            return (
                self.__leading_prompts
                + self.__history_optimizer.optimize_history(self.__history)
                + self.__trailing_prompts
            )

        return self.__leading_prompts + self.__history + self.__trailing_prompts

    def clear_history(self):
        """
        Clears the history. Does not clear the leading and trailing prompts (if any).
        """
        self.__history = []

    def get_history(self):
        """
        Returns the entire message history. Does not include the leading and trailing prompts.

        Do not confuse this method with get_messages, which returns the messages to be sent to the chat agent at a given invocation.

        Returns:
            List[Message]: The list of messages representing the entire history.
        """
        return self.__history
