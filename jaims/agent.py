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
        max_consecutive_tool_calls: int = 10,
    ):
        self.llm_interface = llm_interface
        self.tool_manager = tool_manager or JAImsDefaultToolManager()
        self.tools = tools or []
        self.history_manager = history_manager
        self.max_consecutive_tool_calls = max_consecutive_tool_calls
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

    def __update_session(self, session_messages: List[JAImsMessage]):
        self.__session_iteration += 1
        if self.__session_iteration > self.max_consecutive_tool_calls:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(self.__session_iteration)

        if self.history_manager:
            self.history_manager.add_messages(session_messages)
            self.__session_messages = self.history_manager.get_messages()
        else:
            self.__session_messages.extend(session_messages)

    def __end_session(self, messages: List[JAImsMessage]):

        if self.history_manager:
            self.history_manager.add_messages(messages)

        self.__session_iteration = -1
        self.__session_messages = []

    def __get_tool_results(
        self, message: JAImsMessage, tools: List[JAImsFunctionTool]
    ) -> List[JAImsMessage]:
        tool_results = []
        if message and message.tool_calls:
            tool_call_results = self.tool_manager.handle_tool_calls(
                self, message.tool_calls, tools
            )
            tool_results.extend(tool_call_results)

        return tool_results

    def run(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Runs the agent with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Returns:
            Optional[str]: The response text, or None if there is no response.
        """

        self.__update_session(messages or [])

        run_tools = tools or self.tools

        response_message = self.llm_interface.call(
            self.__session_messages, run_tools, tool_constraints=tool_constraints
        )

        tool_results = self.__get_tool_results(response_message, run_tools)
        if tool_results and not tool_constraints:
            return self.run(
                [response_message] + tool_results,
                run_tools,
                tool_constraints=tool_constraints,
            )

        self.__end_session([response_message] + tool_results)
        return response_message.get_text() or ""

    def run_stream(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Runs the agent in streaming mode with the given messages and tools.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Yields:
            str: The text delta of each response message.
        """

        self.__update_session(messages or [])

        run_tools = tools or self.tools

        streaming_response = self.llm_interface.call_streaming(
            self.__session_messages, run_tools, tool_constraints=tool_constraints
        )

        response_message = None
        for delta_resp in streaming_response:
            response_message = delta_resp.message
            yield delta_resp.textDelta or ""

        if not response_message:
            return

        tool_results = self.__get_tool_results(response_message, run_tools)
        if tool_results and not tool_constraints:
            yield from self.run_stream(
                [response_message] + tool_results,
                run_tools,
                tool_constraints=tool_constraints,
            )
            return

        self.__end_session([response_message] + tool_results)
