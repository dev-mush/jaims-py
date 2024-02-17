from __future__ import annotations
from enum import Enum
import json
from typing import Any, List, Dict, Optional, Callable

from jaims.exceptions import JAImsUnexpectedFunctionCall
from jaims.openai_wrappers import JAImsOptions, JAImsOpenaiKWArgs


# Enum class over all Json Types
class JAImsJsonSchemaType(Enum):
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    ARRAY = "array"
    BOOLEAN = "boolean"
    NULL = "null"


class JAImsParamDescriptor:
    """
    Describes a parameter to be used in the OPENAI API.

    Attributes
    ----------
        name : str
            the parameter name
        description : str
            the parameter description
        json_type : JsonType:
            the parameter json type
        attributes_params_descriptors : list of JAImsParamDescriptor
            the list of parameters descriptors for the attributes of the parameter
            in case the parameter is an object, defualts to None
        array_type_descriptors : list of JAImsParamDescriptor
            the parameter descriptors for the array type in case the parameter is an array, defaults to None
        enum_values:
            the list of values in case the parameter is an enum, defaults to None
        required : bool
            whether the parameter is required or not, defaults to True

    """

    def __init__(
        self,
        name: str,
        description: str,
        json_type: JAImsJsonSchemaType,
        attributes_params_descriptors: Optional[List[JAImsParamDescriptor]] = None,
        array_type_descriptors: Optional[List[JAImsParamDescriptor]] = None,
        array_type_any_valid: bool = True,
        enum_values: Optional[List[Any]] = None,
        required: bool = True,
    ):
        self.name = name
        self.description = description
        self.json_type = json_type
        self.attributes_params_descriptors = attributes_params_descriptors
        self.array_type_descriptors = array_type_descriptors
        self.array_type_any_valid = array_type_any_valid
        self.enum_values = enum_values
        self.required = required

    def get_jsonapi_schema(self) -> Dict[str, Any]:
        """
        Returns the jsonapi schema for the parameter.
        """
        schema: dict[str, Any] = {
            "type": self.json_type.value,
            "description": self.description,
        }

        if (
            self.json_type == JAImsJsonSchemaType.OBJECT
            and self.attributes_params_descriptors
        ):
            schema["properties"] = {}
            schema["required"] = []
            for param in self.attributes_params_descriptors:
                schema["properties"][param.name] = param.get_jsonapi_schema()
                if param.required:
                    schema["required"].append(param.name)

        if self.json_type == JAImsJsonSchemaType.ARRAY and self.array_type_descriptors:
            items_schema = [
                desc.get_jsonapi_schema() for desc in self.array_type_descriptors
            ]
            if self.array_type_any_valid:
                schema["items"] = {"anyOf": items_schema}
            else:
                schema["items"] = [items_schema]

        if self.enum_values:
            schema["enum"] = self.enum_values

        return schema


class JAImsToolDescriptor:
    """
    Describes a tool to be used in the OPENAI API. Supports only function tool for now.

    Attributes
    ----------
        name : str
            the tool name
        description : str
            the tool description
        params_descriptors: List[JAImsParamDescriptor]
            the list of parameters descriptors
    """

    def __init__(
        self,
        name: str,
        description: str,
        params_descriptors: List[JAImsParamDescriptor],
    ):
        self.name = name
        self.description = description
        self.params_descriptors = params_descriptors

    def to_openai_function_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_jsonapi_schema(),
            },
        }

    def get_jsonapi_schema(self) -> Dict[str, Any]:
        """
        Returns the jsonapi schema for function.
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in self.params_descriptors:
            schema["properties"][param.name] = param.get_jsonapi_schema()
            if param.required:
                schema["required"].append(param.name)

        return schema


class JAImsFuncWrapper:
    """
    Wraps a tool used by the OPENAI API along with a function to be called locally when
    the tool is invoked by openai.


    Attributes
    ----------
        function : Callable[..., Any]
            the function to be called
        name : str
            the function name
        description : str
            the function description
        params_descriptors: List[JAImsParamDescriptor]
            the list of parameters descriptors

    Methods
    -------
        call(params: Dict[str, Any]) -> Any
            calls the wrapped function with the given parameters
    """

    def __init__(
        self,
        function: Callable[..., Any],
        function_tool: JAImsToolDescriptor,
    ):
        self.function = function
        self.function_tool = function_tool

    def call(self, params: Dict[str, Any]) -> Any:
        """
        Calls the function with the given parameters.

        Parameters
        ----------
            params : dict
                the parameters to be passed to the function
        """
        return self.function(**params)


class JAImsToolResponse:
    """
    This class offers a way to interact with the agent to trigger events after a tool is called.
    It can be used as a response from a tool call.
    It is not mandatory, return this class from your function tools if you want to interact with the agent to alter the flow of the execution.

    Attributes
    ----------
        content : Any
            the content of the response to be sent to the LLM
        stop: bool
            Whether the tool call should stop the current execution or not, defaults to False.
            This is meant to be used when the tool calling is set to "auto" and it is necessary to
            stop the current execution but, in case of parallel tool calling, all the other tools should be called regardless.
            The net result is that the result of each tool call won't be sent back to the LLM and not tracked in the history.
        halt: bool
            Whether the tool call should stop the current execution or not, defaults to False.
            This is meant to be used when the tool calling is set to "auto" and it is necessary to stop the current execution abruptly.
            This means that in the context of parallel tool calling, as soon as the halt is set to True, all subsequent tool calls are not executed and the current run will terminate.
            The net result is that the result of each tool call won't be sent back to the LLM and not tracked in the history.
        override_kwargs: JAImsOpenaiKWArgs (optional)
            The kwargs to be used to override the current kwargs when giving tool results back to the LLM.
            If parallel tools are called in the same iteration and more than one sets an override_kwargs, the last override_kwargs will be used since, by design, the results are sent back to the LLM in batch.
            Useful for instance to update the model version, the token size or the temperature.
        override_options: JAImsOptions (optional)
            The options to be used to override the current options when giving tool results back to the LLM.
            If parallel tools are called in the same iteration and each sets an override_options, the last override_options will be used since, by design, the results are sent back to the LLM in batch.
            Useful for instance to update the static leading and trailing prompts, finetune the max consecutive calls allowed and so on.
    """

    def __init__(
        self,
        content: Any,
        halt: bool = False,
        override_kwargs: Optional[JAImsOpenaiKWArgs] = None,
        override_options: Optional[JAImsOptions] = None,
    ):
        self.content = content
        self.halt = halt
        self.override_kwargs = override_kwargs
        self.override_options = override_options


class ToolResults:
    """
    Used Internally by the function handler to pass the result back to the agent.

    Attributes
    ----------
        function_result_messages: List[Any]
            the list of function tool result messages to be sent to the LLM
        stop: bool
            Wether the agent should stop the current execution or not, defaults to False.
        override_kwargs: JAImsOpenaiKWArgs (optional)
            Kwargs to be used to override the current kwargs when giving tool results back to the LLM.
        override_options: JAImsOptions (optional)
            Options to be used to override the current options when giving tool results back to the LLM.
    """

    def __init__(
        self,
        function_result_messages: List[Any],
        stop: bool = False,
        override_kwargs: Optional[JAImsOpenaiKWArgs] = None,
        override_options: Optional[JAImsOptions] = None,
    ):
        self.function_result_messages = function_result_messages
        self.stop = stop
        self.override_kwargs = override_kwargs
        self.override_options = override_options


class JAImsFunctionHandler:
    """
    Handles the functions to be used in the OPENAI API.
    Holds a list of JAImsFuncWrapper instances.

    Attributes
    ----------
        functions : List[JAImsFuncWrapper]
            the list of functions to be called
    """

    def handle_from_message(
        self, message: Dict[str, Any], function_wrappers: List[JAImsFuncWrapper]
    ) -> ToolResults:
        """
        Handles a function_call message, calling the appropriate function.

        Parameters
        ----------
            message : dict
                the message from the agent

        Returns
        -------
            dict
                the function result message to be sent to openai

        Raises
        ------
            UnexpectedFunctionCallException
                if the function name is not found in the functions list
        """
        tool_calls = message.get("tool_calls", [])
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
        self, function_calls: List[dict], function_wrappers: List[JAImsFuncWrapper]
    ) -> ToolResults:
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

            fc_result = function_wrapper.call(**fc["args"])
            stop = False
            override_kwargs = None
            override_options = None
            if isinstance(fc_result, JAImsToolResponse):
                if fc_result.halt:
                    return ToolResults(function_result_messages=results, stop=True)

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

        return ToolResults(
            function_result_messages=results,
            stop=stop,
            override_kwargs=override_kwargs,
            override_options=override_options,
        )


def parse_function_wrappers_to_tools(
    func_wrappers: List[JAImsFuncWrapper],
) -> List[Dict[str, Any]]:
    return [
        func_wrapper.function_tool.to_openai_function_tool()
        for func_wrapper in func_wrappers
    ]
