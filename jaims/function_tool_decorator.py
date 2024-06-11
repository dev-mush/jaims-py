from enum import Enum
from jaims.entities import (
    JAImsFunctionTool,
    JAImsFunctionToolDescriptor,
)
from typing import Any, List, Optional, Dict, Type, Union, get_args
from functools import wraps
import inspect
from pydantic import BaseModel, Field, create_model


# def infer_json_type(python_type: Any) -> str:
#     origin_type = getattr(python_type, "__origin__", None)

#     if python_type in {int, float}:
#         return "number"
#     elif python_type is str:
#         return "string"
#     elif python_type is bool:
#         return "boolean"
#     elif origin_type in {list, set, tuple} or python_type in {list, set, tuple}:
#         return "array"
#     elif isinstance(python_type, type) and issubclass(python_type, Enum):
#         return "enum"
#     elif origin_type is dict or python_type is dict:
#         return "dict"
#     elif inspect.isclass(python_type):
#         return "object"
#     else:
#         raise ValueError(f"Unsupported type {python_type}")


# def enum_to_json_schema(enum_class: Any) -> dict:
#     enum_values = [member.value for member in enum_class.__members__.values()]

#     json_types = set()
#     for value in enum_values:
#         if isinstance(value, str):
#             json_types.add("string")
#         elif isinstance(value, (int, float)):
#             json_types.add("number")
#         elif isinstance(value, bool):
#             json_types.add("boolean")
#         elif isinstance(value, list):
#             json_types.add("array")
#         elif isinstance(value, dict):
#             json_types.add("object")
#         else:
#             raise ValueError(f"Unsupported enum value type: {type(value)}")

#     schema = {
#         "type": list(json_types) if len(json_types) > 1 else json_types.pop(),
#         "enum": enum_values,
#     }

#     return schema


# def infer_list_types(input_type: Any) -> List[Type]:

#     args = get_args(input_type)
#     if not args:
#         return []

#     element_type = args[0]

#     if hasattr(element_type, "__origin__") and element_type.__origin__ is Union:
#         return list(get_args(element_type))
#     else:
#         return [element_type]


# def parse_params(
#     params: Any,
# ) -> BaseModel:

#     if params is None:
#         raise ValueError("Params cannot be None")

#     if isinstance(params, BaseModel):
#         return params

#     params_type = type(params)
#     json_type = infer_json_type(params_type)

#     if json_type == "string" or json_type == "number" or json_type == "boolean":
#         params_models = {f"{params_type}_value": (params_type, Field(description=""))}
#         return create_model(
#             "Param", **{params_type.__name__: (params_type, Field(None))}
#         )

#     # checking if the passed descriptor is valid
#     if param_descriptor is not None and not isinstance(param_descriptor, (str, dict)):
#         raise ValueError(
#             f"Invalid descriptor '{param_descriptor}' for {param_type}. Descriptors must be a string or a dictionary"
#         )

#     # if the descriptor is a dictionary, override inferred values with the passed ones
#     if isinstance(param_descriptor, dict):
#         param_name_repr = param_descriptor.get(
#             "name", param_name or param_type.__name__
#         )
#         description = param_descriptor.get("description", "")
#         type_string_repr = param_descriptor.get(
#             "json_type", infer_json_type(param_type)
#         )
#         enum_values_repr = param_descriptor.get("enum", None)
#         required = param_descriptor.get("required", True)
#         attributes_descriptors = param_descriptor.get("attributes", {})
#         array_types_descriptors = param_descriptor.get("array_types", {})

#     else:
#         param_name_repr = param_name or param_type.__name__
#         description = param_descriptor or ""
#         type_string_repr = infer_json_type(param_type)
#         required = True
#         attributes_descriptors = {}
#         array_types_descriptors = {}
#         enum_values_repr = None

#     if type_string_repr == "enum":
#         jsonschema_enum_type = enum_to_json_schema(param_type)
#         return {
#             "name": param_name_repr,
#             "description": description,
#             "type": jsonschema_enum_type["type"],
#             "enum": jsonschema_enum_type["enum"],
#             "required": required,
#         }

#     if type_string_repr == "dict":
#         return {
#             "name": param_name_repr,
#             "description": description,
#             "type": "object",
#             "required": required,
#         }

#     elif type_string_repr == "array":
#         array_types = infer_list_types(param_type)
#         if array_types:
#             array_type_descriptors = []
#             for array_item_type in array_types:

#                 item_type_descriptor = array_types_descriptors.get(
#                     array_item_type.__name__, None
#                 )

#                 array_type_descriptors.append(
#                     parse_param(
#                         param_type=array_item_type,
#                         param_descriptor=item_type_descriptor,
#                     )
#                 )
#             return {
#                 "name": param_name_repr,
#                 "description": description,
#                 "type": "array",
#                 "array_types": array_type_descriptors,
#                 "required": required,
#             }

#     elif type_string_repr == "object":
#         attributes_param_descriptors = []
#         for attribute, value in param_type.__init__.__annotations__.items():
#             attributes_param_descriptors.append(
#                 parse_param(
#                     param_type=value,
#                     param_name=attribute,
#                     param_descriptor=attributes_descriptors.get(attribute, None),
#                 )
#             )
#         return {
#             "name": param_name_repr,
#             "description": description,
#             "type": "object",
#             "attributes": attributes_param_descriptors,
#             "required": required,
#         }

#     return {
#         "name": param_name_repr,
#         "description": description,
#         "type": type_string_repr,
#         "required": required,
#         "enum": enum_values_repr,
#     }


def jaimsfunctiontool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    params_descriptions: Optional[Dict[str, str]] = None,
):
    def decorator(func):
        tool_name = name if name else func.__name__
        tool_description = description or ""
        annotations = func.__annotations__
        params_models = {}

        for param_name, type_ in annotations.items():
            if type_ is inspect._empty:
                raise ValueError(
                    f"Parameter '{param_name}' has no type annotation, please make sure you're type hinting every parameter"
                )
            params_models[param_name] = (
                type_,
                (
                    Field(description=params_descriptions.get(param_name, ""))
                    if params_descriptions
                    else Field()
                ),
            )

        DynamicModel = create_model("DynamicModel", **params_models)

        descriptor = JAImsFunctionToolDescriptor(
            name=tool_name, description=tool_description, params=DynamicModel
        )

        @wraps(func)
        def wrapped(dynamic_model):
            kwargs = {name: getattr(dynamic_model, name) for name in annotations}
            return func(**kwargs)

        return JAImsFunctionTool(
            function=wrapped,
            descriptor=descriptor,
        )

    return decorator
