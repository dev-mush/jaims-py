from abc import ABC, abstractmethod
from typing import List, Generator

from jaims.entities import (
    JAImsMessage,
    JAImsStreamingMessage,
    JAImsToolCall,
    JAImsFunctionTool,
)


class JAImsLLMInterface(ABC):
    @abstractmethod
    def call(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> JAImsMessage:
        raise NotImplementedError

    @abstractmethod
    def call_streaming(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> Generator[JAImsStreamingMessage, None, None]:
        raise NotImplementedError


class JAImsHistoryOptimizer(ABC):
    @abstractmethod
    def optimize_history(self, messages: List[JAImsMessage]) -> List[JAImsMessage]:
        raise NotImplementedError


class JAImsHistoryManager(ABC):
    @abstractmethod
    def add_messages(self, messages: List[JAImsMessage]):
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> List[JAImsMessage]:
        raise NotImplementedError


class JAImsToolManager(ABC):
    @abstractmethod
    def handle_tool_calls(
        self, tool_calls: List[JAImsToolCall], tools: List[JAImsFunctionTool]
    ) -> List[JAImsMessage]:
        raise NotImplementedError
