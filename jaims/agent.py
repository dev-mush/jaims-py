from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import deprecated

from pydantic import BaseModel

if TYPE_CHECKING:
    from .interfaces import LLMAdapterITF, HistoryManagerITF, ToolManagerITF

from typing import Generator, List, Optional


from jaims.default_tool_manager import DefaultToolManager


from jaims.entities import (
    FunctionToolDescriptor,
    MaxConsecutiveFunctionCallsExceeded,
    Message,
    StreamingMessage,
    FunctionTool,
    LLMParams,
    Config,
    ModelT,
)

supported_providers_list = [
    "openai",
    "google",
    "mistral",
    "anthropic",
    "vertex",
]

SUPPORTED_PROVIDERS = Literal[
    "openai",
    "google",
    "mistral",
    "anthropic",
    "vertex",
]


class _SessionState(Enum):
    RUN = auto()
    COMPLETE = auto()
    ERROR = auto()


class _SessionStateController:
    def __init__(
        self,
        initial_messages: List[Message],
        tools: List[FunctionTool],
        max_consecutive_tool_call_retries: int,
        max_consecutive_iterations: int,
        tool_manager: ToolManagerITF,
        tool_call_error_behavior: str,
        tool_constraints: Optional[List[str]] = None,
        history_manager: Optional[HistoryManagerITF] = None,
    ):
        self.state = _SessionState.RUN
        self.__error = None
        self._iterations = -1
        self.__tool_call_retries = -1
        self.__tools = tools
        self.__tool_constraints = tool_constraints
        self.__tool_call_behavior = tool_call_error_behavior
        self.__history_manager = history_manager
        self.__tool_manager = tool_manager
        self.__max_consecutive_tool_call_retries = max_consecutive_tool_call_retries
        self.__max_consecutive_iterations = max_consecutive_iterations
        self.__initial_messages = initial_messages
        self.__session_messages = []
        self.__update_messages(initial_messages)

    def get_next_iteration_messages(self) -> List[Message]:
        if self.__history_manager:
            return self.__history_manager.get_messages()

        return self.__session_messages

    def get_session_response_messages(self) -> List[Message]:
        return self.__session_messages[len(self.__initial_messages) :]

    def get_error(self) -> Any:
        return self.__error

    def __update_messages(self, messages: List[Message]):
        self.__session_messages.extend(messages)
        if self.__history_manager:
            self.__history_manager.add_messages(messages)

    def __should_forward_to_llm(self, messages: List[Message]) -> bool:
        """
        Evaluates if the messages contain any data that should be forwarded to the LLM
        depending on the current tool_constraints configuration.
        """
        if self.__tool_constraints and len(self.__tool_constraints) > 0:
            return False

        for message in messages:
            if message.tool_calls:
                return True

        return False

    def update(
        self,
        llm_response: Message,
    ):
        # check initial state and rule out illegal state transitions
        if self.state == _SessionState.ERROR:
            raise Exception(
                "Attempting to update a session in error state. Error: ", self.__error
            )

        if self.state == _SessionState.COMPLETE:
            raise Exception("Attempting to update a session in complete state.")

        next_iteration_messages = [llm_response]

        # evaluate current iteration error state
        self._iterations += 1
        if self._iterations > self.__max_consecutive_iterations:
            self.state = _SessionState.ERROR
            self.__error = MaxConsecutiveFunctionCallsExceeded(self._iterations)
            return

        # evaluate tool calls request
        if llm_response.tool_calls:
            self.__tool_call_retries += 1
            if self.__tool_call_retries > self.__max_consecutive_tool_call_retries:
                self.state = _SessionState.ERROR
                self.__error = MaxConsecutiveFunctionCallsExceeded(
                    self.__tool_call_retries
                )
                return

            tool_responses = self.__tool_manager.handle_tool_calls(
                llm_response.tool_calls,
                self.__tools,
            )

            errors = [tr for tr in tool_responses if tr.is_error]
            if errors:
                if self.__tool_call_behavior == "retry":
                    # retry the whole iteration by only increasing the iteration counter
                    return
                elif self.__tool_call_behavior == "raise":
                    errors_str = "\n".join(
                        [f"{e.tool_name}: {e.response}" for e in errors]
                    )
                    self.state = _SessionState.ERROR
                    self.__error = Exception(
                        f"Error while calling tools:\n {errors_str}"
                    )
                    return

            next_iteration_messages.append(
                Message.tool_response_message(tool_responses)
            )

        self.__update_messages(next_iteration_messages)

        if self.__should_forward_to_llm(next_iteration_messages):
            self.state = _SessionState.RUN

        else:
            self.state = _SessionState.COMPLETE


class Agent:
    """
    Base  JAIms Agent class, interacts with the LLM Interface to run messages and tools.

    Tools can be injected in the constructor once, or passed when invoking (overriding the injected tools at every invocation).
    Provide an history_manager if you plan to use the agent as a chat agent and want to keep track of the conversation.
    Provide a tool_manager if you want to customize the tool handling.

    Args:
        llm_interface (LLMIITF): The LLM interface to use.
        tools (Optional[List[FunctionTool]]): The list of tools to use. Defaults to None.
        history_manager (Optional[HistoryManager]): The history manager to use. Defaults to None.
        tool_manager (Optional[ToolManager]): The tool manager to use. Defaults to None.
        tools: Optional[List[FunctionTool]]): The list of tools to use. Defaults to None.
        tool_call_error_behavior (Literal["retry", "raise", "forward_to_llm"]): The behavior to adopt when a tool call fails. Defaults to "retry".retry will retry the tool call, raise will raise an exception, forward_to_llm will forward the error message to the LLM.
        max_consecutive_tool_calls (int): The maximum number of consecutive tool calls allowed. Defaults to 10, raise an exception if exceeded.
        max_consecutive_tool_call_retries (int): The maximum number of consecutive retries allowed when a tool call fails (client side) and tool_call_error_behavir is "retry". Defaults to 3, raise an exception if exceeded.
    """

    def __init__(
        self,
        llm_interface: LLMAdapterITF,
        history_manager: Optional[HistoryManagerITF] = None,
        tool_manager: Optional[ToolManagerITF] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
        max_consecutive_tool_calls: int = 10,
        max_consecutive_tool_call_retries: int = 3,
        tool_call_error_behavior: Literal["retry", "raise", "forward_to_llm"] = "retry",
    ):
        self.llm_interface = llm_interface
        self.tool_manager = tool_manager or DefaultToolManager()
        self.tools = tools or []
        self.history_manager = history_manager
        self.max_consecutive_tool_calls = max_consecutive_tool_calls
        self.max_consecutive_tool_call_retries = max_consecutive_tool_call_retries
        self.tool_constraints = tool_constraints
        self.tool_call_error_behavior = tool_call_error_behavior

    @staticmethod
    def build(
        model: str,
        provider: SUPPORTED_PROVIDERS,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        llm_params: Optional[LLMParams] = None,
        history_manager: Optional[HistoryManagerITF] = None,
        tool_manager: Optional[ToolManagerITF] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Agent:
        """
        Factory method to build an agent with the specified parameters.

        This method is the recommended way to create an agent instance, you can directly specify the model and provider to use, and pass the most common configuration here. If you need fine-grained control over the agent creation, you can create the agent instance manually by instantiating the wanted LLMAdapterITF (or implementing your own) and passing it to the Agent constructor.

        Remember to install the required dependencies for the provider you want to use, e.g. for OpenAI, Google, and Mistral:

        ```bash
        pip install jaims-py[openai, google, mistral]
        ```

        The API key will be read from the default environment variables when not provided.

        Args:
            model (str): The model to use.
            provider (SUPPORTED_PROVIDERS): The provider to use.
            api_key (Optional[str]): The API key. Defaults to None.
            config (Optional[Config]): The config to use. Defaults to None. (When none, the default config will be used).
            llm_params (Optional[LLMParams]): The llm_params to be used. Defaults to None. (When none, the default llm_params will be used).
            history_manager (Optional[HistoryManager]): The history manager. Defaults to None. Use DefaultHistoryManager to implement chat-like behavior.
            tool_manager (Optional[ToolManager]): The tool manager. Defaults to None. Usually you don't need to provide a custom tool manager.
            tools (Optional[List[FunctionTool]]): The list of tools. Defaults to None. You can pass tools here so they are defined just once, or pass them when invoking the agent for each run.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. Defaults to None. When None, the LLM works in agent mode (if supported) and picks the tools to use, otherwise it will use the tools that match the constraints. (Depends on the implementation of the LLM Interface).

        Returns:
            Agent: The agent instance.
        """

        assert (
            provider in supported_providers_list
        ), f"currently supported providers are: [{', '.join(supported_providers_list)}]. If you're targeting an unsupported provider you should supply your own adapter instead."

        if provider == "openai":
            from .factories import openai_factory

            return openai_factory(
                model=model,
                api_key=api_key,
                config=config,
                llm_params=llm_params,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
                tool_constraints=tool_constraints,
            )
        elif provider == "google":
            from .factories import google_factory

            return google_factory(
                model=model,
                api_key=api_key,
                config=config,
                llm_params=llm_params,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
                tool_constraints=tool_constraints,
            )
        elif provider == "mistral":
            from .factories import mistral_factory

            return mistral_factory(
                model=model,
                api_key=api_key,
                config=config,
                llm_params=llm_params,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
                tool_constraints=tool_constraints,
            )
        elif provider == "anthropic":
            from .factories import anthropic_factory

            return anthropic_factory(
                model=model,
                api_key=api_key,
                config=config,
                llm_params=llm_params,
                history_manager=history_manager,
                tool_manager=tool_manager,
                tools=tools,
                tool_constraints=tool_constraints,
            )

        elif provider == "vertex":
            if "claude" in model:

                from .factories import anthropic_factory

                return anthropic_factory(
                    model=model,
                    api_key=api_key,
                    provider="vertex",
                    config=config,
                    llm_params=llm_params,
                    history_manager=history_manager,
                    tool_manager=tool_manager,
                    tools=tools,
                    tool_constraints=tool_constraints,
                )

            else:
                from .factories import vertex_ai_factory

                return vertex_ai_factory(
                    model=model,
                    api_key=api_key,
                    config=config,
                    llm_params=llm_params,
                    history_manager=history_manager,
                    tool_manager=tool_manager,
                    tools=tools,
                    tool_constraints=tool_constraints,
                )

    def run(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> List[Message]:
        """
        Runs the agent with the given messages and tools and returns the response message.

        When the agent has been instantiated with tools and tool_constraints and this method receives tools and tool_constraints, the tools and tool_constraints passed to this method will override the ones injected in the constructor, only for this run (they get discarded after the run and those injected in the constructor will be used again).

        Args:
            messages (Optional[List[Message]]): The list of messages. Defaults to None.
            tools (Optional[List[FunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Returns:
            List[Message]: The response messages.
        """

        messages = messages or []
        run_tools = tools or self.tools
        tool_constraints = tool_constraints or self.tool_constraints

        session_controller = _SessionStateController(
            initial_messages=messages,
            tools=run_tools,
            max_consecutive_tool_call_retries=self.max_consecutive_tool_call_retries,
            max_consecutive_iterations=self.max_consecutive_tool_calls,
            tool_constraints=tool_constraints,
            tool_manager=self.tool_manager,
            history_manager=self.history_manager,
            tool_call_error_behavior=self.tool_call_error_behavior,
        )

        while session_controller.state == _SessionState.RUN:
            llm_response = self.llm_interface.call(
                session_controller.get_next_iteration_messages(),
                run_tools,
                tool_constraints=tool_constraints,
            )

            session_controller.update(llm_response)

        if session_controller.state == _SessionState.ERROR:
            raise session_controller.get_error()

        return session_controller.get_session_response_messages()

    @staticmethod
    def run_model(
        model: str,
        provider: SUPPORTED_PROVIDERS,
        api_key: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tools_constraints: Optional[List[str]] = None,
        tool_manager: Optional[ToolManagerITF] = None,
        config: Optional[Config] = None,
        llm_params: Optional[LLMParams] = None,
    ) -> List[Message]:
        """
        Runs the specified model with the given parameters and returns the response message. Doesn't require an Agent instance.

        This method is useful for inline usage when you don't need to keep the agent instance around. Has the same effect of creating an agent instance and calling the run method.

        Args:
            model (str): The model to use.
            provider (Literal): The provider to use (among the supported providers).
            api_key (Optional[str]): The API key. Defaults to None.
            messages (Optional[List[Message]]): The list of messages to be sent to the LLM. Defaults to None.
            tools (Optional[List[FunctionTool]]): The list of tools that the LLM will use. Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.
            tool_manager (Optional[ToolManagerITF]): The tool manager. Defaults to None. Usually you don't need to provide a custom tool manager.
            config (Optional[Config]): The config to use. Defaults to None. (When none, the default config will be used).
            llm_params (Optional[LLMParams]): The llm_params to be used. Defaults to None. (When none, the default llm_params will be used).

        Returns:
            Message: The response message.
        """

        agent = Agent.build(
            model=model,
            provider=provider,
            api_key=api_key,
            config=config,
            llm_params=llm_params,
            tool_manager=tool_manager,
            tools=tools,
        )

        return agent.run(messages=messages, tool_constraints=tools_constraints)

    def run_tool(
        self,
        descriptor: FunctionToolDescriptor[ModelT],
        messages: Optional[List[Message]] = None,
    ) -> ModelT:
        """
        Runs a single tool with the given messages and returns the expected response data defined by the tool descriptor.

        This method is useful when you want to execute a task with an LLM and want the result formatted into a pydantic model to be used in your application.

        Args:
            descriptor (FunctionToolDescriptor): The tool descriptor to use.
            messages (Optional[List[Message]]): The list of messages. Defaults to None.

        Returns:
            ModelT: The expected response data defined by the tool descriptor.
        """

        response_data = None

        def callback(response: BaseModel):
            nonlocal response_data
            response_data = response

        tool = FunctionTool(
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
        descriptor: FunctionToolDescriptor[ModelT],
        messages: Optional[List[Message]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        llm_params: Optional[LLMParams] = None,
        tool_manager: Optional[ToolManagerITF] = None,
    ) -> ModelT:
        """
        Runs a single tool with the given messages and returns the expected response data defined by the tool descriptor. Doesn't require an Agent instance.

        This method is useful for inline usage when you don't need to keep the agent instance around. Has the same effect of creating an agent instance and calling the run_tool method.

        Args:
            model (str): The model to use.
            provider (Literal): The provider to use (among the supported providers).
            descriptor (FunctionToolDescriptor): The tool descriptor to use.
            messages (Optional[List[Message]]): The list of messages. Defaults to None.
            api_key (Optional[str]): The API key. Defaults to None.
            config (Optional[Config]): The config to use. Defaults to None. (When none, the default config will be used).
            llm_params (Optional[LLMParams]): The llm_params to be used. Defaults to None. (When none, the default llm_params will be used).
            tool_manager (Optional[ToolManagerITF]): The tool manager. Defaults to None. Usually you don't need to provide a custom tool manager.

        Returns:
            ModelT: The expected response data defined by the tool descriptor.
        """

        agent = Agent.build(
            model=model,
            provider=provider,
            api_key=api_key,
            config=config,
            llm_params=llm_params,
            tool_manager=tool_manager,
        )

        return agent.run_tool(descriptor, messages)

    def run_stream(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[StreamingMessage, None, None]:
        """
        Runs the agent in streaming mode with the given messages and tools and yields the streaming response message.

        Usually this method is used when you're implementing a chat-like interface that has function calling capabilities, and you want to stream the response messages as they come.

        When the agent has been instantiated with tools and tool_constraints and this method receives tools and tool_constraints, the tools and tool_constraints passed to this method will override the ones injected in the constructor, only for this run (they get discarded after the run and those injected in the constructor will be used again).

        Args:
            messages (Optional[List[Message]]): The list of messages. Defaults to None.
            tools (Optional[List[FunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Yields:
            StreamingMessage: The streaming response message.
        """

        messages = messages or []
        run_tools = tools or self.tools
        tool_constraints = tool_constraints or self.tool_constraints

        session_controller = _SessionStateController(
            initial_messages=messages,
            tools=run_tools,
            max_consecutive_tool_call_retries=self.max_consecutive_tool_call_retries,
            max_consecutive_iterations=self.max_consecutive_tool_calls,
            tool_constraints=tool_constraints,
            tool_manager=self.tool_manager,
            history_manager=self.history_manager,
            tool_call_error_behavior=self.tool_call_error_behavior,
        )

        while session_controller.state == _SessionState.RUN:
            streaming_response = self.llm_interface.call_streaming(
                session_controller.get_next_iteration_messages(),
                run_tools,
                tool_constraints=tool_constraints,
            )

            response_message = None
            for delta_resp in streaming_response:
                response_message = delta_resp.message
                yield delta_resp

            if not response_message:
                return

            session_controller.update(response_message)

        if session_controller.state == _SessionState.ERROR:
            raise session_controller.get_error()

    def message(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Sends the messages to the agent returning the response text, in a chat-like fashion.

        This method is useful when implementing a chat-bot, since you can use the response text directly. Has the same effect of calling the run method and concatenating the response messages.

        Args:
            messages (Optional[List[Message]]): The list of messages. Defaults to None.
            tools (Optional[List[FunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Returns:
            Optional[str]: The response text, or None if there is no response.
        """

        messages = self.run(messages, tools, tool_constraints)
        if not messages:
            return None

        full_response = ""
        for message in messages:
            text_message = message.get_text()
            if text_message:
                full_response += f"\n{text_message}"

        return full_response

    def message_stream(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Sends the messages to the agent returning the response text in a chat-like fashion, in a streaming fashion.

        This method is useful when implementing a chat-bot, since you can use the response text directly. Has the same effect of calling the run_stream method and concatenating the response messages.

        Args:
            messages (Optional[List[Message]]): The list of messages. Defaults to None.
            tools (Optional[List[FunctionTool]]): When passed, the tools will override any tools injected in the constructor, only for this run. When None, the tools injected in the constructor will be used (if any). Defaults to None.
            tool_constraints (Optional[List[str]]): The list of tool identifiers that should be used. When None, the LLM works in agent mode (if supported) and picks the tools to use. Defaults to None.

        Yields:
            str: The text delta of each response message.
        """

        for delta_resp in self.run_stream(messages, tools, tool_constraints):
            yield delta_resp.textDelta or ""


@deprecated("use Agent instead")
class JAImsAgent(Agent):
    pass
