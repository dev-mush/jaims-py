from enum import Enum
from jaims.entities import (
    JAImsJsonSchemaType,
    JAImsFunctionTool,
    JAImsFunctionToolDescriptor,
)
from typing import Any, List, Optional, Dict, Type, Union, get_args
import collections.abc
from functools import wraps
import inspect


def infer_json_type(python_type: Any) -> str:
    origin_type = getattr(python_type, "__origin__", None)

    if python_type in {int, float}:
        return "number"
    elif python_type is str:
        return "string"
    elif python_type is bool:
        return "boolean"
    elif not python_type or python_type is type(None):
        return "null"
    elif origin_type in {list, set, tuple} or python_type in {list, set, tuple}:
        return "array"
    elif isinstance(python_type, type) and issubclass(python_type, Enum):
        return "enum"
    elif origin_type is dict or python_type is dict:
        return "dict"
    elif inspect.isclass(python_type):
        return "object"
    else:
        raise ValueError(f"Unsupported type {python_type}")


def enum_to_json_schema(enum_class: Any) -> dict:
    enum_values = [member.value for member in enum_class.__members__.values()]

    json_types = set()
    for value in enum_values:
        if isinstance(value, str):
            json_types.add("string")
        elif isinstance(value, (int, float)):
            json_types.add("number")
        elif isinstance(value, bool):
            json_types.add("boolean")
        elif value is None:
            json_types.add("null")
        elif isinstance(value, list):
            json_types.add("array")
        elif isinstance(value, dict):
            json_types.add("object")
        else:
            raise ValueError(f"Unsupported enum value type: {type(value)}")

    schema = {
        "type": list(json_types) if len(json_types) > 1 else json_types.pop(),
        "enum": enum_values,
    }

    return schema


def infer_list_types(input_type: Any) -> List[Type]:

    args = get_args(input_type)
    if not args:
        return []

    element_type = args[0]

    if hasattr(element_type, "__origin__") and element_type.__origin__ is Union:
        return list(get_args(element_type))
    else:
        return [element_type]


def format_param_descriptor_dict(
    param_type: Any,
    param_name: Optional[str] = None,
    param_descriptor: Optional[Union[str, Dict]] = None,
) -> Dict[str, Any]:

    # checking if the passed descriptor is valid
    if param_descriptor is not None and not isinstance(param_descriptor, (str, dict)):
        raise ValueError(
            f"Invalid descriptor '{param_descriptor}' for {param_type}. Descriptors must be a string or a dictionary"
        )

    # if the descriptor is a dictionary, override inferred values with the passed ones
    if isinstance(param_descriptor, dict):
        param_name_repr = param_descriptor.get(
            "name", param_name or param_type.__name__
        )
        description = param_descriptor.get("description", "")
        type_string_repr = param_descriptor.get(
            "json_type", infer_json_type(param_type)
        )
        enum_values_repr = param_descriptor.get("enum", None)
        required = param_descriptor.get("required", True)
        attributes_descriptors = param_descriptor.get("attributes", {})
        array_types_descriptors = param_descriptor.get("array_types", {})

    else:
        param_name_repr = param_name or param_type.__name__
        description = param_descriptor or ""
        type_string_repr = infer_json_type(param_type)
        required = True
        attributes_descriptors = {}
        array_types_descriptors = {}
        enum_values_repr = None

    if type_string_repr == "enum":
        jsonschema_enum_type = enum_to_json_schema(param_type)
        return {
            "name": param_name_repr,
            "description": description,
            "type": jsonschema_enum_type["type"],
            "enum": jsonschema_enum_type["enum"],
            "required": required,
        }

    if type_string_repr == "dict":
        return {
            "name": param_name_repr,
            "description": description,
            "type": "object",
            "required": required,
        }

    elif type_string_repr == "array":
        array_types = infer_list_types(param_type)
        if array_types:
            array_type_descriptors = []
            for array_item_type in array_types:

                item_type_descriptor = array_types_descriptors.get(
                    array_item_type.__name__, None
                )

                array_type_descriptors.append(
                    format_param_descriptor_dict(
                        param_type=array_item_type,
                        param_descriptor=item_type_descriptor,
                    )
                )
            return {
                "name": param_name_repr,
                "description": description,
                "type": "array",
                "array_types": array_type_descriptors,
                "required": required,
            }

    elif type_string_repr == "object":
        attributes_param_descriptors = []
        for attribute, value in param_type.__init__.__annotations__.items():
            attributes_param_descriptors.append(
                format_param_descriptor_dict(
                    param_type=value,
                    param_name=attribute,
                    param_descriptor=attributes_descriptors.get(attribute, None),
                )
            )
        return {
            "name": param_name_repr,
            "description": description,
            "type": "object",
            "attributes": attributes_param_descriptors,
            "required": required,
        }

    return {
        "name": param_name_repr,
        "description": description,
        "type": type_string_repr,
        "required": required,
        "enum": enum_values_repr,
    }


def jaimsfunctiontool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_descriptors: Optional[Dict[str, Any]] = None,
):
    """
    [Experimental]

    Decorator to create a JAImsFunctionToolDescriptor from a function.

    The decorator uses the following syntax to parse the param_descriptors attributes to generate descriptors for the function parameters:

    - Each key of the dictionary matches the name of the parameter in the function signature (or the name of the attribute in the objects), in this case the value can be:
      - a string used as description of the parameter for primitive types, the json type will be inferred with reflection.
      - a dictionary for object types, with string keys for each attribute of the object and values that follow the same rules.

    Use the following optional keys to define the parameter descriptor:
        - description : the description of the parameter, when not passed defaults to empty string
        - json_type: when not passed it is inferred with reflection
        - required: whether the parameter is required or not, defaults to True
        - attributes: a dictionary with the attributes of the object, following the same syntax rules (so either string or dictionary)
        - enum: a list of values in case the parameter is an enum, defaults to None
        - array_types: a dictionary with the descriptors of the array types, following the same syntax rules (so either string or dictionary)

    Args:
        name: the name of the tool, if None the function name will be used
        description: the description of the tool, defaults to empty string
        param_descriptors: a dictionary with the descriptors of the parameters, following the syntax rules described above.
    """

    def decorator(func):
        sig = inspect.signature(func)
        tool_name = name if name else func.__name__
        passed_descriptors = param_descriptors or {}
        raw_descriptors = []

        for param_name, param in sig.parameters.items():
            if param.annotation is inspect._empty:
                raise ValueError(
                    f"Parameter '{param_name}' has no type annotation, please make sure you're type hinting every parameter"
                )

            raw_descriptors.append(
                format_param_descriptor_dict(
                    param_type=param.annotation,
                    param_name=param_name,
                    param_descriptor=passed_descriptors.get(param_name, None),
                )
            )

        func_tool_raw_descriptor = {
            "name": tool_name,
            "description": description or "",
            "params": raw_descriptors,
        }

        @wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        return JAImsFunctionTool(
            function=wrapped,
            function_tool_descriptor=JAImsFunctionToolDescriptor.from_dict(
                func_tool_raw_descriptor
            ),
        )

    return decorator
