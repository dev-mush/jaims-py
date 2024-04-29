from typing import Generator, List, Optional


from jaims.interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager
from jaims.default_history_manager import JAImsDefaultHistoryManager
from jaims.default_tool_manager import JAImsDefaultToolManager


from jaims.entities import (
    JAImsMaxConsecutiveFunctionCallsExceeded,
    JAImsMessage,
    JAImsFunctionTool,
)

# TODO: Abstract messages and history Manager
# TODO: Refactor all docstrings
# TODO: Leading Prompts and Trailing Prompts are misplaced in the options, should think about it
# TODO: Might ditch the expense counter. It's not really useful and it's not accurate.
# TODO: Transaction storage is platform specific, should be abstracted away
# TODO: Check roles of tool calls and responses on openai
# TODO: Refactor logging entirely


class JAImsAgent:

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
        self.__history_manager = history_manager or JAImsDefaultHistoryManager()

    # TODO: Decide return type, might not be ideal to return a string
    def run(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        iteration_n: int = 0,
        max_iterations: int = 10,
    ) -> Optional[str]:

        if iteration_n >= max_iterations:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(iteration_n)

        messages = messages or []
        self.__history_manager.add_messages(messages)
        current_window_messages = self.__history_manager.get_messages()
        response = self.__llm_interface.call(current_window_messages, self.__tools)

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

        if iteration_n >= max_iterations:
            raise JAImsMaxConsecutiveFunctionCallsExceeded(iteration_n)

        messages = messages or []
        self.__history_manager.add_messages(messages)
        current_window_messages = self.__history_manager.get_messages()

        response = self.__llm_interface.call_streaming(
            current_window_messages, self.__tools
        )

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

    @staticmethod
    def __merge_tool_calls(existing_tool_calls, new_tool_calls_delta):
        if not existing_tool_calls:
            return new_tool_calls_delta

        new_tool_calls = existing_tool_calls[:]
        for new_call_delta in new_tool_calls_delta:
            existing_call = next(
                (item for item in new_tool_calls if item.index == new_call_delta.index),
                None,
            )
            if not existing_call:
                new_tool_calls.append(new_call_delta)
            else:
                if (
                    existing_call.type != new_call_delta.type
                    and new_call_delta.type is not None
                ):
                    existing_call.type = new_call_delta.type
                if (
                    existing_call.id != new_call_delta.id
                    and new_call_delta.id is not None
                ):
                    existing_call.id = new_call_delta.id
                if existing_call.function is None:
                    existing_call.function = new_call_delta.function
                else:
                    if (
                        existing_call.function.name != new_call_delta.function.name
                        and new_call_delta.function.name is not None
                    ):
                        existing_call.function.name = new_call_delta.function.name
                    existing_call.function.arguments = (
                        existing_call.function.arguments or ""
                    ) + (new_call_delta.function.arguments or "")

        return new_tool_calls

    @staticmethod
    def __accumulate_choice_delta(accumulator, new_delta):
        if accumulator is None:
            return new_delta

        if new_delta.content:
            accumulator.content = (accumulator.content or "") + new_delta.content
        if new_delta.role:
            accumulator.role = (accumulator.role or "") + new_delta.role
        if new_delta.tool_calls:
            accumulator.tool_calls = JAImsAgent.__merge_tool_calls(
                accumulator.tool_calls, new_delta.tool_calls
            )

        return accumulator
