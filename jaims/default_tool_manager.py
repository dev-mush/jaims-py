from __future__ import annotations
from typing import TYPE_CHECKING


from typing import List
from jaims.entities import (
    JAImsFunctionTool,
    JAImsUnexpectedFunctionCall,
    JAImsToolCall,
    JAImsToolResponse,
)

from jaims.interfaces import JAImsToolManager


class JAImsDefaultToolManager(JAImsToolManager):
    """
    A class that manages tool calls in JAIms.

    This class handles tool calls by executing the corresponding wrapped functions and formatting the results as messages to be consumed by the LLM.
    Supports parallel execution of multiple tool calls.

    Attributes:
        None

    Methods:
        handle_tool_calls: Executes the tool calls and formats the results as messages to be consumed by the LLM.

    """

    def handle_tool_calls(
        self,
        tool_calls: List[JAImsToolCall],
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsToolResponse]:
        """
        Executes the tool calls and formats the results as messages to be consumed by the LLM.

        The results of the tools have to be json-serializable in order to be sent as messages.

        Args:
            tool_calls (List[JAImsToolCall]): The list of tool calls to be executed.
            tools (List[JAImsFunctionTool]): The list of available tools.

        Returns:
            List[JAImsToolResponse]: A list of JAImsToolResponse objects representing the results of the tool calls.

        Raises:
            JAImsUnexpectedFunctionCall: If a tool call does not match any available tool.

        """
        results = []
        for fc in tool_calls:
            function_name = fc.tool_name
            tool_wrapper = next(
                (f for f in tools if f.descriptor.name == function_name),
                None,
            )
            if not tool_wrapper:
                raise JAImsUnexpectedFunctionCall(function_name)

            args = fc.tool_args or {}
            try:
                fc_result = tool_wrapper.call_raw(**args)
                results.append(
                    JAImsToolResponse(
                        tool_call_id=fc.id,
                        tool_name=fc.tool_name,
                        response=fc_result,
                    )
                )
            except Exception as e:
                results.append(
                    JAImsToolResponse(
                        tool_call_id=fc.id,
                        tool_name=fc.tool_name,
                        response=str(e),
                        is_error=True,
                    )
                )

        return results
