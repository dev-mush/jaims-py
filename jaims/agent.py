from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager

from typing import Generator, List, Optional


from jaims.default_tool_manager import JAImsDefaultToolManager


from jaims.entities import (
    JAImsMaxConsecutiveFunctionCallsExceeded,
    JAImsMessage,
    JAImsFunctionTool,
    JAImsLLMConfig,
    JAImsOptions,
)


# TODOS:
# High Priority
# TODO: Refactor all docstrings and docs, check imports and remove unused imports

# Mid Priority
# TODO: Implement a method to pass a function directly instead of the jaims descriptors
# TODO: Implement Tests

# Low Priority
# TODO: Add support for multiple completion responses
# TODO: Adjust transaction storage for openai streaming response
# TODO: Refactor logging entirely


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

    @staticmethod
    def build(
        model: str,
        provider: Literal["openai", "google"],
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        config: Optional[JAImsLLMConfig] = None,
        history_manager: Optional[JAImsHistoryManager] = None,
        tool_manager: Optional[JAImsToolManager] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
    ) -> JAImsAgent:

        # assert provider in ["openai", "google"], "Provider must be either 'openai' or 'google'"
        assert provider in [
            "openai",
            "google",
        ], f"curretnly supported providers are: [openai, google] . If you're targeting an unsupported provider you should supply your own adapter instead."

        if provider == "openai":
            from .factories import openai_factory

            return openai_factory(
                model=model,
                api_key=api_key,
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )
        elif provider == "google":
            from .factories import google_factory

            return google_factory(
                model=model,
                api_key=api_key,
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )
        else:
            raise ValueError("Provider is not supported.")

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
