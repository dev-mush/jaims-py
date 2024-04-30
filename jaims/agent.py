from typing import Generator, List, Optional


from jaims.interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager
from jaims.default_history_manager import JAImsDefaultHistoryManager
from jaims.default_tool_manager import JAImsDefaultToolManager


from jaims.entities import (
    JAImsMaxConsecutiveFunctionCallsExceeded,
    JAImsMessage,
    JAImsFunctionTool,
)

# TODO: Refactor all docstrings
# TODO: Refactor logging entirely
# TODO: Decide return type of runs, might not be ideal to return a string
# TODO: Implement a method to pass a function directly instead of the jaims descriptors

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
        self.__llm_interface = llm_interface
        self.__tool_manager = tool_manager or JAImsDefaultToolManager()
        self.__tools = tools or []
        self.__history_manager = history_manager

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

        if self.__history_manager:
            self.__history_manager.add_messages(session_messages)
            session_messages = self.__history_manager.get_messages()

        return session_messages

    def run(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        iteration_n: int = 0,
        max_iterations: int = 10,
    ) -> Optional[str]:
        """
        Runs the agent with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            iteration_n (int): The current iteration number. Defaults to 0.
            max_iterations (int): The maximum number of iterations. Defaults to 10. Pass this to constrain the maximum number of function calls.

        Returns:
            Optional[str]: The response text, or None if there is no response.
        """
        if iteration_n >= max_iterations:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(iteration_n)

        session_messages = self.__get_messages_for_run(messages)

        response = self.__llm_interface.call(session_messages, self.__tools)

        if response.tool_calls:
            tool_response_messages = self.__tool_manager.handle_tool_calls(
                response.tool_calls, self.__tools
            )
            if tool_response_messages:
                return self.run(tool_response_messages, iteration_n + 1, max_iterations)

        return response.text or ""

    def run_stream(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        iteration_n: int = 0,
        max_iterations: int = 10,
    ) -> Generator[str, None, None]:
        """
        Runs the agent in streaming mode with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            iteration_n (int): The current iteration number. Defaults to 0.
            max_iterations (int): The maximum number of iterations. Defaults to 10. Pass this to constrain the maximum number of function calls.

        Yields:
            str: The text delta of each response message.
        """
        if iteration_n >= max_iterations:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(iteration_n)

        session_messages = self.__get_messages_for_run(messages)

        response = self.__llm_interface.call_streaming(session_messages, self.__tools)

        response_message = None
        for delta_resp in response:
            response_message = delta_resp.message
            yield delta_resp.textDelta or ""

        if response_message and response_message.tool_calls:
            tool_response_messages = self.__tool_manager.handle_tool_calls(
                response_message.tool_calls, self.__tools
            )
            if tool_response_messages:
                yield from self.run_stream(
                    tool_response_messages, iteration_n + 1, max_iterations
                )
