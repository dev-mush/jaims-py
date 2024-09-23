from __future__ import annotations


from typing import List
from jaims.entities import (
    FunctionTool,
    UnexpectedFunctionCall,
    ToolCall,
    ToolResponse,
)

from jaims.interfaces import ToolManagerITF


class DefaultToolManager(ToolManagerITF):
    """
    Manages tool calls in JAIms.

    This class handles tool calls by executing the corresponding wrapped functions and formatting the results as messages to be consumed by the LLM.
    Supports parallel execution of multiple tool calls.

    Attributes:
        None

    Methods:
        handle_tool_calls: Executes the tool calls and formats the results as messages to be consumed by the LLM.

    """

    def handle_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tools: List[FunctionTool],
    ) -> List[ToolResponse]:
        """
        Executes the tool calls and formats the results as messages to be consumed by the LLM.

        Args:
            tool_calls (List[ToolCall]): The list of tool calls to be executed.
            tools (List[FunctionTool]): The function tool wrappers to be used to execute the tool calls.

        Returns:
            List[ToolResponse]: A list of ToolResponse objects representing the results of the tool calls.

        Raises:
            UnexpectedFunctionCall: If a tool call is not found in the list of function tools.

        """
        results = []
        for fc in tool_calls:
            function_name = fc.tool_name
            tool_wrapper = next(
                (f for f in tools if f.descriptor.name == function_name),
                None,
            )
            if not tool_wrapper:
                raise UnexpectedFunctionCall(function_name)

            args = fc.tool_args or {}
            try:
                fc_result = tool_wrapper.call_raw(**args)
                results.append(
                    ToolResponse(
                        tool_call_id=fc.id,
                        tool_name=fc.tool_name,
                        response=fc_result,
                    )
                )
            except Exception as e:
                results.append(
                    ToolResponse(
                        tool_call_id=fc.id,
                        tool_name=fc.tool_name,
                        response=str(e),
                        is_error=True,
                    )
                )

        return results
