from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager

from typing import Generator, List, Optional


from jaims.default_tool_manager import JAImsDefaultToolManager


from jaims.entities import (
    JAImsMaxConsecutiveFunctionCallsExceeded,
    JAImsMessage,
    JAImsFunctionTool,
)

# TODO: Refactor all docstrings
# TODO: Refactor logging entirely
# TODO: Adjust imports in __init__.py files making them more explicit
# TODO: Adjust transaction storage for openai streaming response
# TODO: Add original response to jaims message and return it instead of strings
# TODO: Decide return type of runs, might not be ideal to return a string
# TODO: Implement a method to pass a function directly instead of the jaims descriptors
# TODO: Add images support

# TODO: Test Stuff:
# - Agent:
#  - with and without history
#  - with and without tools
#  - use mocks for the interfaces
# - History Managers []
# - Tool Manager []
#


class JAImsAgent:
    """
    Base  JAIms Agent class, interacts with the JAImsLLMInterface to run messages and tools.

    Args:
        llm_interface (JAImsLLMInterface): The LLM interface used for communication.
        history_manager (Optional[JAImsHistoryManager]): The history manager to track session messages. Defaults to None, which means no history is kept and messages are lost after run.
        tool_manager (Optional[JAImsToolManager]): The tool manager to handle tool calls. if None, the Default tool manager is used. Implement your own interface for advanced scenarios.
        tools (Optional[List[JAImsFunctionTool]]): The list of function tools. Defaults to None.

    Methods:
        run: Runs the agent with the given messages and tools.
        run_stream: Runs the agent in streaming mode with the given messages and tools.
    """

    def __init__(
        self,
        llm_interface: JAImsLLMInterface,
        history_manager: Optional[JAImsHistoryManager] = None,
        tool_manager: Optional[JAImsToolManager] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
    ):
        self.llm_interface = llm_interface
        self.tool_manager = tool_manager or JAImsDefaultToolManager()
        self.tools = tools or []
        self.history_manager = history_manager
        self.__session_iteration = -1
        self.__session_messages = []

    def __get_messages_for_run(
        self, messages: Optional[List[JAImsMessage]]
    ) -> List[JAImsMessage]:
        """
        Processes a list of messages for the agent run and, if necessary, adds them to history.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.

        Returns:
            List[JAImsMessage]: The session messages.
        """
        session_messages = messages or []

        if self.history_manager:
            self.history_manager.add_messages(session_messages)
            session_messages = self.history_manager.get_messages()

        return session_messages

    def __update_session(
        self, session_messages: List[JAImsMessage], max_iterations: int
    ):
        self.__session_iteration += 1
        if self.__session_iteration > max_iterations:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(self.__session_iteration)

        if self.history_manager:
            self.history_manager.add_messages(session_messages)
            self.__session_messages = self.history_manager.get_messages()
        else:
            self.__session_messages.extend(session_messages)

    def __end_session(self, response: Optional[JAImsMessage] = None):

        if self.history_manager and response:
            self.history_manager.add_messages([response])

        self.__session_iteration = -1
        self.__session_messages = []

    def __evaluate_tool_results(self, message: Optional[JAImsMessage]):
        tool_results = []
        if message and message.tool_calls:
            tool_call_results = self.tool_manager.handle_tool_calls(
                self, message.tool_calls, self.tools
            )
            tool_results = [message] + tool_call_results

        return tool_results

    def run(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        max_iterations: int = 10,
    ) -> Optional[str]:
        """
        Runs the agent with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            max_iterations (int): The maximum number of iterations. Defaults to 10. Pass this to constrain the maximum number of function calls.

        Returns:
            Optional[str]: The response text, or None if there is no response.
        """

        self.__update_session(messages or [], max_iterations)

        response_message = self.llm_interface.call(self.__session_messages, self.tools)

        tool_results = self.__evaluate_tool_results(response_message)
        if tool_results:
            return self.run(tool_results, max_iterations)

        self.__end_session(response_message)
        return response_message.get_text() or ""

    def run_stream(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        max_iterations: int = 10,
    ) -> Generator[str, None, None]:
        """
        Runs the agent in streaming mode with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            max_iterations (int): The maximum number of iterations. Defaults to 10. Pass this to constrain the maximum number of function calls.

        Yields:
            str: The text delta of each response message.
        """

        self.__update_session(messages or [], max_iterations)

        streaming_response = self.llm_interface.call_streaming(
            self.__session_messages, self.tools
        )

        response_message = None
        for delta_resp in streaming_response:
            response_message = delta_resp.message
            yield delta_resp.textDelta or ""

        tool_results = self.__evaluate_tool_results(response_message)
        if tool_results:
            yield from self.run_stream(tool_results, max_iterations)
            return

        self.__end_session(response_message)
