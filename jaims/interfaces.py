from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import JAImsAgent

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
        self,
        agent: JAImsAgent,
        tool_calls: List[JAImsToolCall],
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsMessage]:
        raise NotImplementedError
