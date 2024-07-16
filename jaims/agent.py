from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from .interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager

from typing import Generator, List, Optional


from jaims.default_tool_manager import JAImsDefaultToolManager


from jaims.entities import (
    JAImsFunctionToolDescriptor,
    JAImsMaxConsecutiveFunctionCallsExceeded,
    JAImsMessage,
    JAImsStreamingMessage,
    JAImsFunctionTool,
    JAImsLLMConfig,
    JAImsOptions,
)

supported_providers_list = [
    "openai",
    "google",
    "mistral",
    "anthropic",
    "vertex",
    "bedrock",
]
SUPPORTED_PROVIDERS = Literal[
    "openai", "google", "mistral", "anthropic", "vertex", "bedrock"
]


class JAImsAgent:
    """
    Base  JAIms Agent class, interacts with the JAImsLLMInterface to run messages and tools.

    Tools can be injected in the constructor once, or passed when invoking (overriding the injected tools at every invocation).
    Provide an history_manager if you plan to use the agent as a chat agent and want to keep track of the conversation.
    Provide a tool_manager if you want to customize the tool handling.

    Args:
        llm_interface (JAImsLLMInterface): The LLM interface to use.
        tools (Optional[List[JAImsFunctionTool]]): The list of tools to use. Defaults to None.
        history_manager (Optional[JAImsHistoryManager]): The history manager to use. Defaults to None.
        tool_manager (Optional[JAImsToolManager]): The tool manager to use. Defaults to None.
        max_consecutive_tool_calls (int): The maximum number of consecutive tool calls allowed. Defaults to 10.
    """

    def __init__(
        self,
        llm_interface: JAImsLLMInterface,
        history_manager: Optional[JAImsHistoryManager] = None,
        tool_manager: Optional[JAImsToolManager] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        max_consecutive_tool_calls: int = 10,
        tool_call_error_behavior: Literal[
            "raise", "forward_to_llm", "ignore"
        ] = "raise",
    ):
        self.llm_interface = llm_interface
        self.tool_manager = tool_manager or JAImsDefaultToolManager()
        self.tools = tools or []
        self.history_manager = history_manager
        self.max_consecutive_tool_calls = max_consecutive_tool_calls
        self.__session_iteration = -1
        self.__session_messages = []
        self.tool_call_error_behavior = tool_call_error_behavior

    @staticmethod
    def build(
        model: str,
        provider: SUPPORTED_PROVIDERS,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        config: Optional[JAImsLLMConfig] = None,
        history_manager: Optional[JAImsHistoryManager] = None,
        tool_manager: Optional[JAImsToolManager] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
    ) -> JAImsAgent:
        """
        Factory method to build an agent with the specified parameters.

        Currently available providers are: [openai, google, mistral]. Make sure to install the required dependencies using:

        ```bash
        pip install jaims-py[openai, google, mistral]
        ```

        The API key will be read from the default environment variables when not provided.

        Args:
            model (str): The model to use.
            provider (SUPPORTED_PROVIDERS): The provider to use.
            api_key (Optional[str]): The API key. Defaults to None.
            options (Optional[JAImsOptions]): The options. Defaults to None.
            config (Optional[JAImsLLMConfig]): The config. Defaults to None.
            history_manager (Optional[JAImsHistoryManager]): The history manager. Defaults to None.
            tool_manager (Optional[JAImsToolManager]): The tool manager. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): The list of tools. Defaults to None.

        Returns:
            JAImsAgent: The agent instance.
        """

        assert (
            provider in supported_providers_list
        ), f"currently supported providers are: [{', '.join(supported_providers_list)}]. If you're targeting an unsupported provider you should supply your own adapter instead."

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
        elif provider == "mistral":
            from .factories import mistral_factory

            return mistral_factory(
                model=model,
                api_key=api_key,
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )
        elif provider == "anthropic":
            from .factories import anthropic_factory

            return anthropic_factory(
                model=model,
                api_key=api_key,
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )

        elif provider == "vertex":
            if "claude" not in model:
                raise ValueError(
                    "The vertex provider is only available for anthropic models for now."
                )

            from .factories import anthropic_factory

            return anthropic_factory(
                model=model,
                api_key=api_key,
                provider="vertex",
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )

        elif provider == "bedrock":
            if "claude" not in model:
                raise ValueError(
                    "The bedrock provider is only available for anthropic models for now."
                )

            from .factories import anthropic_factory

            return anthropic_factory(
                model=model,
                api_key=api_key,
                provider="bedrock",
                options=options,
                config=config,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
            )

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

    def __should_forward_to_llm(
        self, messages: List[JAImsMessage], tool_constraints: Optional[List[str]] = None
    ) -> bool:
        """
        Evaluates if the messages contain any data that should be forwarded to the LLM
        depending on the current tool_constraints configuration.
        """
        if tool_constraints and len(tool_constraints) > 0:
            return False

        for message in messages:
            if message.tool_calls:
                return True

        return False

    def __evaluate_response_messages(
        self,
        llm_response_message: JAImsMessage,
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsMessage]:
        """
        Evaluates the response message and returns a list of messages to be sent to the LLM, possibly including tool results.
        """
        final_messages = [llm_response_message]

        if llm_response_message.tool_calls:
            tool_responses = self.tool_manager.handle_tool_calls(
                self,
                llm_response_message.tool_calls,
                tools,
            )
            if tool_responses:
                final_messages.append(
                    JAImsMessage.tool_response_message(tool_responses)
                )

        return final_messages

    def run(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> JAImsMessage:
        """
        Runs the agent with the given messages and tools and returns the response message.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Returns:
            JAImsMessage: The response message.
        """

        self.__update_session(messages or [])

        run_tools = tools or self.tools

        response_message = self.llm_interface.call(
            self.__session_messages, run_tools, tool_constraints=tool_constraints
        )

        next_messages = self.__evaluate_response_messages(response_message, run_tools)

        if self.__should_forward_to_llm(next_messages, tool_constraints):
            return self.run(
                next_messages,
                run_tools,
                tool_constraints=tool_constraints,
            )

        self.__end_session(next_messages)
        return response_message

    @staticmethod
    def run_model(
        model: str,
        provider: Literal[
            "openai", "google", "mistral", "anthropic", "vertex", "bedrock"
        ],
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tools_constraints: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        config: Optional[JAImsLLMConfig] = None,
        tool_manager: Optional[JAImsToolManager] = None,
    ) -> JAImsMessage:
        """
        Runs the specified model with the given parameters and returns the response message.

        Args:
            model (str): The model to use.
            provider (Literal["openai", "google", "mistral", "anthropic","vertex"]): The provider to use.
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): The list of tools. Defaults to None.
            tools_constraints (Optional[List[str]]): The list of tool identifiers that should be used. Defaults to None.
            api_key (Optional[str]): The API key. Defaults to None.
            options (Optional[JAImsOptions]): The options. Defaults to None.
            config (Optional[JAImsLLMConfig]): The config. Defaults to None.
            tool_manager (Optional[JAImsToolManager]): The tool manager. Defaults to None.

        Returns:
            JAImsMessage: The response message.
        """

        agent = JAImsAgent.build(
            model=model,
            provider=provider,
            api_key=api_key,
            options=options,
            config=config,
            tool_manager=tool_manager,
            tools=tools,
        )

        return agent.run(messages=messages, tool_constraints=tools_constraints)

    def run_tool(
        self,
        descriptor: JAImsFunctionToolDescriptor,
        messages: Optional[List[JAImsMessage]] = None,
    ) -> Any:
        """
        Runs a single tool with the given messages and returns the expected response data.

        Args:
            tool (JAImsFunctionToolDescriptor): The tool to run.
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.

        Returns:
            BaseModel: The expected response data defined by the tool descriptor.
        """

        response_data = None

        def callback(response: BaseModel):
            nonlocal response_data
            response_data = response

        tool = JAImsFunctionTool(
            descriptor=descriptor,
            function=callback,
        )

        self.run(messages, [tool], tool_constraints=[descriptor.name])

        if not response_data:
            raise ValueError(f"Tool {tool.descriptor.name} did not return any data.")

        return response_data

    @staticmethod
    def run_tool_model(
        model: str,
        provider: SUPPORTED_PROVIDERS,
        descriptor: JAImsFunctionToolDescriptor,
        messages: Optional[List[JAImsMessage]] = None,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        config: Optional[JAImsLLMConfig] = None,
        tool_manager: Optional[JAImsToolManager] = None,
    ) -> Any:
        """
        Runs a single tool with the given messages and returns the expected response data.

        Args:
            model (str): The model to use.
            provider (Literal["openai", "google", "mistral", "anthropic","vertex"]): The provider to use.
            descriptor (JAImsFunctionToolDescriptor): The tool to run.
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            api_key (Optional[str]): The API key. Defaults to None.
            options (Optional[JAImsOptions]): The options. Defaults to None.
            config (Optional[JAImsLLMConfig]): The config. Defaults to None.
            tool_manager (Optional[JAImsToolManager]): The tool manager. Defaults to None.

        Returns:
            BaseModel: The expected response data defined by the tool descriptor.
        """

        agent = JAImsAgent.build(
            model=model,
            provider=provider,
            api_key=api_key,
            options=options,
            config=config,
            tool_manager=tool_manager,
        )

        return agent.run_tool(descriptor, messages)

    def run_stream(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[JAImsStreamingMessage, None, None]:
        """
        Runs the agent in streaming mode with the given messages and tools and yields the streaming response message.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Yields:
            JAImsStreamingMessage: The streaming response message.
        """

        self.__update_session(messages or [])

        run_tools = tools or self.tools

        streaming_response = self.llm_interface.call_streaming(
            self.__session_messages, run_tools, tool_constraints=tool_constraints
        )

        response_message = None
        for delta_resp in streaming_response:
            response_message = delta_resp.message
            yield delta_resp

        if not response_message:
            return

        next_messages = self.__evaluate_response_messages(response_message, run_tools)

        if self.__should_forward_to_llm(next_messages, tool_constraints):
            yield from self.run_stream(
                next_messages,
                run_tools,
                tool_constraints=tool_constraints,
            )
            return

        self.__end_session(next_messages)

    def message(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Sends the messages to the agent returning the response text, in a chat-like fashion.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Returns:
            Optional[str]: The response text, or None if there is no response.
        """

        message = self.run(messages, tools, tool_constraints)

        return message.get_text() if message else None

    def message_stream(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Sends the messages to the agent and streams the response text, in a chat-like fashion.

        Args:
            messages (Optional[List[JAImsMessage]]): The list of messages. Defaults to None.
            tools (Optional[List[JAImsFunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Yields:
            str: The text delta of each response message.
        """

        for delta_resp in self.run_stream(messages, tools, tool_constraints):
            yield delta_resp.textDelta or ""
