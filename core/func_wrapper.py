from enum import Enum
from typing import Any, List, Dict, Optional, Callable, Union


# Enum class over all Json Types
class JsonSchemaType(Enum):
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
        array_type_descriptor : JAImsParamDescriptor
            the parameter descriptor for the array type in case the parameter is an array, defaults to None
        enum_values:
            the list of values in case the parameter is an enum, defaults to None
        required : bool
            whether the parameter is required or not, defaults to True

    """

    def __init__(
        self,
        name: str,
        description: str,
        json_type: JsonSchemaType,
        attributes_params_descriptors: Optional[List["JAImsParamDescriptor"]] = None,
        array_type_descriptor: Optional["JAImsParamDescriptor"] = None,
        enum_values: Optional[List[Any]] = None,
        required: bool = True,
    ):
        self.name = name
        self.description = description
        self.json_type = json_type
        self.attributes_params_descriptors = attributes_params_descriptors
        self.array_type_descriptor = array_type_descriptor
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
            self.json_type == JsonSchemaType.OBJECT
            and self.attributes_params_descriptors
        ):
            schema["properties"] = {}
            schema["required"] = []
            for param in self.attributes_params_descriptors:
                schema["properties"][param.name] = param.get_jsonapi_schema()
                if param.required:
                    schema["required"].append(param.name)

        if self.json_type == JsonSchemaType.ARRAY and self.array_type_descriptor:
            schema["items"] = {"type": self.array_type_descriptor.json_type.value}

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
