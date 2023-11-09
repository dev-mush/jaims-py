from __future__ import annotations
from enum import Enum
import json
from typing import Any, List, Dict, Optional, Callable

from jaims.exceptions import JAImsUnexpectedFunctionCall


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


class JAImsFuncWrapper:
    """
    Wraps a function call to be used in the OPENAI API.
    Holds the function to be called as well as the function name, its description that the agent will understand,
    and the response json schema.

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
            calls the function with the given parameters
        get_jsonapi_schema() -> Dict[str, Any]
            returns the jsonapi schema for the function
    """

    def __init__(
        self,
        function: Callable[..., Any],
        name: str,
        description: str,
        params_descriptors: List[JAImsParamDescriptor],
    ):
        self.function = function
        self.name = name
        self.description = description
        self.params_descriptors = params_descriptors

    def call(self, params: Dict[str, Any]) -> Any:
        """
        Calls the function with the given parameters.

        Parameters
        ----------
            params : dict
                the parameters to be passed to the function
        """
        return self.function(**params)

    def get_jsonapi_schema(self) -> Dict[str, Any]:
        """
        Returns the jsonapi schema for the function.
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
        self, message: Dict[str, Any], functions: List[JAImsFuncWrapper]
    ) -> List[Dict[str, Any]]:
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
        return self.__call_functions(function_calls, functions)

    def __call_functions(
        self, function_calls: List[dict], functions
    ) -> List[dict[str, Any]]:
        # Check if function_name exists in functions, if not, raise UnexpectedFunctionCallException

        results = []
        for fc in function_calls:
            function_name = fc["name"]
            function_wrapper = next(
                (f for f in functions if f.name == function_name), None
            )
            if not function_wrapper:
                raise JAImsUnexpectedFunctionCall(function_name)

            try:
                fc_result = function_wrapper.function(**fc["args"])
                results.append(
                    {
                        "name": function_name,
                        "tool_call_id": fc["tool_call_id"],
                        "role": "tool",
                        "content": json.dumps(fc_result),
                    }
                )
            except TypeError as e:
                results.append(
                    {
                        "name": function_name,
                        "tool_call_id": fc["tool_call_id"],
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}),
                    }
                )

        # If the name of the current function matches the provided name
        # Call the function and return its result
        return results


def parse_function_wrappers_to_tools(
    functions: List[JAImsFuncWrapper],
) -> List[Dict[str, Any]]:
    openai_functions = []
    for function in functions:
        function_data = {
            k: v
            for k, v in {
                "name": function.name,
                "description": function.description,
                "parameters": function.get_jsonapi_schema(),
            }.items()
            if v is not None
        }

        openai_functions.append({"type": "function", "function": function_data})

    return openai_functions
