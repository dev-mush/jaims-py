from typing import List, Optional

from jaims.entities import JAImsMessage
from jaims.interfaces import JAImsHistoryManager, JAImsHistoryOptimizer


class JAImsLastNHistoryOptimizer(JAImsHistoryOptimizer):
    def __init__(self, last_n: int):
        self.__last_n = last_n

    def optimize_history(self, messages: List[JAImsMessage]) -> List[JAImsMessage]:
        return messages[-self.__last_n :] if self.__last_n > 0 else []


class JAImsDefaultHistoryManager(JAImsHistoryManager):
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
        self.__history.extend(messages)

    def get_messages(self):

        if self.__history_optimizer:
            return (
                self.__leading_prompts
                + self.__history_optimizer.optimize_history(self.__history)
                + self.__trailing_prompts
            )

        return self.__leading_prompts + self.__history + self.__trailing_prompts

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history = []

    def get_history(self):
        """
        Returns entire history.
        """
        return self.__history
