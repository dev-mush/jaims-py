from typing import List, Optional

from jaims.entities import JAImsMessage
from jaims.interfaces import JAImsHistoryManager, JAImsHistoryOptimizer


class JAImsLastNHistoryOptimizer(JAImsHistoryOptimizer):
    """
    A history optimizer that keeps only the last N messages in the history.
    """

    def __init__(self, last_n: int):
        self.__last_n = last_n

    def optimize_history(self, messages: List[JAImsMessage]) -> List[JAImsMessage]:
        """
        Optimizes the history by keeping only the last N messages.

        Args:
            messages (List[JAImsMessage]): The list of messages to optimize.

        Returns:
            List[JAImsMessage]: The optimized list of messages.
        """
        return messages[-self.__last_n :] if self.__last_n > 0 else []


class JAImsDefaultHistoryManager(JAImsHistoryManager):
    """
    A class that handles and manages the history of JAIms messages when used as a chat-agent.

    Use the attribtutes leading_prompts and trailing_prompts to add prompts that will be added to the beginning and end of the history respectively.

    The method get_messages retrieves the messages in the history that will be sent to the chat agent, optionally optimized by the history optimizer.

    Attributes:
        history (Optional[List[JAImsMessage]]): The list of JAIms messages representing the history.
        leading_prompts (Optional[List[JAImsMessage]]): The list of JAIms messages representing the leading prompts.
        trailing_prompts (Optional[List[JAImsMessage]]): The list of JAIms messages representing the trailing prompts.
        history_optimizer (Optional[JAImsHistoryOptimizer]): The history optimizer used to optimize the history.

    Methods:
        add_messages(messages: List[JAImsMessage]): Adds a list of JAIms messages to the history.
        get_messages(): Retrieves the messages in the history, optionally optimized by the history optimizer.
        clear_history(): Clears the history.
        get_history(): Returns the entire history.
    """

    def __init__(
        self,
        history: Optional[List[JAImsMessage]] = None,
        leading_prompts: Optional[List[JAImsMessage]] = None,
        trailing_prompts: Optional[List[JAImsMessage]] = None,
        history_optimizer: Optional[JAImsHistoryOptimizer] = None,
    ):
        self.__history = history or []
        self.__leading_prompts = leading_prompts or []
        self.__trailing_prompts = trailing_prompts or []
        self.__history_optimizer = history_optimizer

    def add_messages(self, messages: List[JAImsMessage]):
        """
        Adds a list of JAIms messages to the history.

        Args:
            messages (List[JAImsMessage]): The list of JAIms messages to be added to the history.
        """
        self.__history.extend(messages)

    def get_messages(self):
        """
        Retrieves the messages in the history, along with the leading and trailing prompts, optionally optimized by the history optimizer.

        Returns:
            List[JAImsMessage]: The list of JAIms messages in the history.
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
        Clears the history. Does not clear the leading and trailing prompts.
        """
        self.__history = []

    def get_history(self):
        """
        Returns the entire message history.

        Returns:
            List[JAImsMessage]: The list of JAIms messages representing the entire history.
        """
        return self.__history
