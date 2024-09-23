from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

from abc import ABC, abstractmethod
from typing import List, Generator, Optional


from jaims.entities import (
    Message,
    StreamingMessage,
    ToolCall,
    FunctionTool,
    ToolResponse,
)


class LLMAdapterITF(ABC):
    """
    Interface of a LLM Adapter.

    Implement this interface to create a custom adapter for a Language Learning Model (LLM) to be used by the JAIms Agent.

    Methods:
        call: Invoked by the Agent instance to call the LLM with a list of messages, tools and tool constraints, returns the output message.
        call_streaming: Invoked by the Agent instance to call the LLM with a list of messages, tools and tool constraints, returns a generator of streaming messages.
    """

    @abstractmethod
    def call(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Message:
        """
        Executes the language learning model on a list of messages and tools.

        Override this method to implement the logic for calling the LLM with a list of messages and tools.

        Args:
            messages: A list of Message objects representing the input messages to be sent to the LLM. Defaults to None.
            tools: A list of FunctionTool objects representing the tools to be applied. Defaults to None.
            tool_constraints: An optional list of tool constraints. Defaults to None.

        Returns:
            A Message object representing the output message from the LLM.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def call_streaming(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[StreamingMessage, None, None]:
        """
        Executes the language learning model on a list of messages and tools, returning a generator of streaming messages.

        Override this method to implement the logic for calling the LLM with a list of messages and tools, returning a generator of streaming messages.

        Args:
            messages: A list of Message objects representing the input messages to be sent to the LLM. Defaults to None.
            tools: A list of FunctionTool objects representing the tools to be applied. Defaults to None.
            tool_constraints: An optional list of tool constraints. Defaults to None.

        Returns:
            A generator of StreamingMessage objects representing the output messages from the LLM.
        """
        raise NotImplementedError


class HistoryOptimizerITF(ABC):
    """
    Interface of a history optimizer supported by the Default HistoryManager.

    Implement this interface for custom history optimization logic (e.g., reducing messages based on token consideration, timestamps, etc.).

    Methods:
        optimize_history: Receives the entire history and returns the sublist based on optimization logic.
    """

    @abstractmethod
    def optimize_history(self, messages: List[Message]) -> List[Message]:
        """
        Optimizes the history based on the optimization logic.

        Args:
            messages (List[Message]): The entire history to be optimized.

        Returns:
            List[Message]: The optimized history based on the optimization logic.
        """
        raise NotImplementedError


class HistoryManagerITF(ABC):
    """
    Interface of a History Manager.

    Implement this interface to create a custom history manager for the JAIms Agent.

    Methods:
        add_messages: Adds a list of JAIms messages to the history.
        get_messages: Retrieves the list of JAIms messages intended to be sent, at a specific invocation, to the LLM.
    """

    @abstractmethod
    def add_messages(self, messages: List[Message]):
        """
        Adds a list of messages to the history.

        Override this method to implement the logic for adding a list of messages to the history. This method is invoked by the Agent instance when a run method is called.

        Args:
            messages (List[Message]): The list of messages to add to the history.
        """
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """
        Returns the list of messages to be sent to the LLM at a given invocation.

        Override this method to implement the logic for retrieving the list of messages to be sent to the LLM at a given invocation. This method is invoked by the Agent instance when a run method is called.

        Returns:
            List[Message]: The list of messages to be sent to the LLM at a given invocation.
        """
        raise NotImplementedError


class ToolManagerITF(ABC):
    """
    Interface of a Tool Manager.

    Usually you don't need to implement this class because it is the core of this library, but in case you need to implement your own logic for handling tool calls, you can do it by subclassing this class.

    Methods:
        handle_tool_calls: Called by the agent to execute the tool calls and format the results as messages to be consumed by the LLM.
    """

    @abstractmethod
    def handle_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tools: List[FunctionTool],
    ) -> List[ToolResponse]:
        """
        Handles the tool calls with on the available FunctionTools.

        When overriding this method, usually you want to inspect the id of the incoming tool_calls, and check them against the avialable tools (if any). Then for each tool call, you will return the corresponding ToolResponse.

        Args:
            tool_calls (List[ToolCall]): List of tool calls to be executed.
            tools (List[FunctionTool]): List of available FunctionTools.

        Returns:
            List[ToolResponse]: List of ToolResponse objects representing the results of the tool calls.
        """
        raise NotImplementedError
