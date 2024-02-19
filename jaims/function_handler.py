from __future__ import annotations
import json
from typing import Any, List, Dict
from jaims.entities import (
    JAImsFuncWrapper,
    JAImsFunctionToolResponse,
    JAImsToolResults,
    JAImsUnexpectedFunctionCall,
)


class JAImsToolHandler:
    """
    Handler delegate class that receives the tool calls requests from the agent and calls the appropriate function wrappers.
    Subclass this class to implement your own function handler in case you need to handle the tool calls in a different way.

    """

    def handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        function_wrappers: List[JAImsFuncWrapper],
    ) -> JAImsToolResults:
        """
        Called by the agent when it is necessary to handle tool calls.

        Parameters
        ----------
            tool_calls : List[Dict[str, Any]]
                the list of tool calls passed by the agent that needs to be handled.
            function_wrappers : List[JAImsFuncWrapper]
                the list of function wrappers relative to the tool calls.

        Returns
        -------
            ToolResults
                the result of the tools executions

        Raises
        ------
            UnexpectedFunctionCallException
                if the function name is not found in the functions list
        """
        function_calls = []
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            function_calls.append(
                {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "args": json.loads(function_args) if function_args else {},
                }
            )

        # invoke function
        return self.__call_functions(function_calls, function_wrappers)

    def __call_functions(
        self,
        function_calls: List[dict],
        function_wrappers: List[JAImsFuncWrapper],
    ) -> JAImsToolResults:
        # Check if function_name exists in functions, if not, raise UnexpectedFunctionCallException

        results = []
        for fc in function_calls:
            function_name = fc["name"]
            function_wrapper = next(
                (f for f in function_wrappers if f.function_tool.name == function_name),
                None,
            )
            if not function_wrapper:
                raise JAImsUnexpectedFunctionCall(function_name)

            fc_result = function_wrapper.call(fc["args"])
            stop = False
            override_kwargs = None
            override_options = None
            if isinstance(fc_result, JAImsFunctionToolResponse):
                if fc_result.halt:
                    return JAImsToolResults(function_result_messages=results, stop=True)

                stop = fc_result.halt
                override_kwargs = fc_result.override_kwargs
                override_options = fc_result.override_options
                fc_result = fc_result.content

            results.append(
                {
                    "name": function_name,
                    "tool_call_id": fc["tool_call_id"],
                    "role": "tool",
                    "content": json.dumps(fc_result),
                }
            )

        return JAImsToolResults(
            function_result_messages=results,
            stop=stop,
            override_kwargs=override_kwargs,
            override_options=override_options,
        )
