from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import JAImsAgent

from typing import List
from jaims.entities import (
    JAImsFunctionTool,
    JAImsUnexpectedFunctionCall,
    JAImsToolCall,
    JAImsMessage,
)

from jaims.interfaces import JAImsToolManager


class JAImsDefaultToolManager(JAImsToolManager):

    def handle_tool_calls(
        self,
        agent: JAImsAgent,
        tool_calls: List[JAImsToolCall],
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsMessage]:
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
            fc_result = tool_wrapper.call_raw(**args)
            results.append(
                JAImsMessage.tool_response_message(
                    tool_call_id=fc.id,
                    tool_name=fc.tool_name,
                    response=fc_result,
                )
            )

        return results
