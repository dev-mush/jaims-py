from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import JAImsAgent

from abc import ABC, abstractmethod
from typing import List, Generator, Optional


from jaims.entities import (
    JAImsMessage,
    JAImsStreamingMessage,
    JAImsToolCall,
    JAImsFunctionTool,
    JAImsToolResponse,
)


class JAImsLLMInterface(ABC):
    """
    Abstract base class for JAImsLLMInterface.

    This class defines the interface for a JAIms LLM (Language Learning Model) interface.
    Subclasses of this class must implement the `call` and `call_streaming` methods.

    Attributes:
        None

    Methods:
        call: Executes the language learning model on a list of messages and tools.
        call_streaming: Executes the language learning model on a list of messages and tools in a streaming fashion.
    """

    @abstractmethod
    def call(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> JAImsMessage:
        """
        Executes the language learning model on a list of messages and tools.

        Args:
            messages: A list of JAImsMessage objects representing the input messages. Defaults to None.
            tools: A list of JAImsFunctionTool objects representing the tools to be applied. Defaults to None.
            tool_constraints: An optional list of tool constraints. Defaults to None.

        Returns:
            A JAImsMessage object representing the output message.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def call_streaming(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[JAImsStreamingMessage, None, None]:
        """
        Executes the language learning model on a list of messages and tools in a streaming fashion.

        Args:
            messages: A list of JAImsMessage objects representing the input messages. Defaults to None.
            tools: A list of JAImsFunctionTool objects representing the tools to be applied. Defaults to None.
            tool_constraints: An optional list of tool constraints. Defaults to None.

        Yields:
            A JAImsStreamingMessage object representing the output message.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class JAImsHistoryOptimizer(ABC):
    """
    Abstract base class for a history optimizer supported by the Default JAImsHistoryManager.

    Override this class to implement custom history optimization logic (e.g., reducing messages based on token consideration, timestamps, etc.).
    """

    @abstractmethod
    def optimize_history(self, messages: List[JAImsMessage]) -> List[JAImsMessage]:
        """
        Optimize the given list of JAIms messages.

        Parameters:
        - messages (List[JAImsMessage]): The list of JAIms messages to optimize.

        Returns:
        - List[JAImsMessage]: The optimized list of JAIms messages.
        """
        raise NotImplementedError


class JAImsHistoryManager(ABC):
    """
    Abstract base class for managing the history of JAIms messages.

    The intended use, when implementing it is to use the add_messages method to push messages in the conversation history, and the get_messages method to retrieve the messages intended to be sent to the LLM when it is invoked. Ideally you might want to send just a slice of the conversation history to the LLM, and not the entire history, to lower token consumption.
    """

    @abstractmethod
    def add_messages(self, messages: List[JAImsMessage]):
        """
        Adds a list of JAIms messages to the history.

        Args:
            messages (List[JAImsMessage]): The list of JAIms messages to add.
        """
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> List[JAImsMessage]:
        """
        Retrieves the list of JAIms messages intended to be sent, at a specific invocation, to the LLM.

        In other words this method should return "a slice" of the conversation history when the user sends a message.

        Returns:
            List[JAImsMessage]: The list of JAIms messages.
        """
        raise NotImplementedError


class JAImsToolManager(ABC):
    """
    Abstract base class for managing JAIms tools.

    This class defines the interface for handling tool calls in JAIms.

    Attributes:
        agent (JAImsAgent): The JAIms agent associated with the tool calls.
        tool_calls (List[JAImsToolCall]): List of tool calls.
        tools (List[JAImsFunctionTool]): List of available tools.

    Returns:
        List[JAImsMessage]: List of messages generated by the tool calls.
    """

    @abstractmethod
    def handle_tool_calls(
        self,
        tool_calls: List[JAImsToolCall],
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsToolResponse]:
        """
        Handle tool calls in JAIms.

        This method should be implemented by subclasses to handle the tool calls.
        The list of tool calls represents a request by the LLM to execute a set of tools, the tools are the actual function wrappers that should be executed.

        Args:
            agent (JAImsAgent): The JAIms agent.
            tool_calls (List[JAImsToolCall]): List of tool calls requested by the LLM.
            tools (List[JAImsFunctionTool]): List of available tools.

        Returns:
            List[JAImsToolResponse]: List of tool responses generated by the tool calls.
        """
        raise NotImplementedError
