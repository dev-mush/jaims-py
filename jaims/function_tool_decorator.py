from enum import Enum
from jaims.entities import (
    JAImsFunctionTool,
    JAImsFunctionToolDescriptor,
)
from typing import Any, List, Optional, Dict, Type, Union, get_args
from functools import wraps
import inspect
from pydantic import BaseModel, Field, create_model


def jaimsfunctiontool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    params_descriptions: Optional[Dict[str, str]] = None,
    return_value_object_name: str = "ToolResponse",
):
    def decorator(func):
        tool_name = name if name else func.__name__
        tool_description = description or ""
        params_models = {}

        sig = inspect.signature(func)
        is_method = "self" in sig.parameters or "cls" in sig.parameters

        for param_name, param in sig.parameters.items():

            # skip self, cls and return
            if param_name in ["self", "cls", "return"]:
                continue

            if param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{param_name}' has no type annotation, please make sure you're type hinting every parameter"
                )

            param_description = (
                params_descriptions.get(param_name, "") if params_descriptions else ""
            )

            if param.default is inspect._empty:
                params_models[param_name] = (
                    param.annotation,
                    Field(description=param_description),
                )
            else:
                params_models[param_name] = (
                    param.annotation,
                    Field(default=param.default, description=param_description),
                )

        if len(params_models) == 1 and issubclass(
            list(params_models.values())[0][0], BaseModel
        ):
            output_model = list(params_models.values())[0][0]
        else:
            output_model = create_model(return_value_object_name, **params_models)

        descriptor = JAImsFunctionToolDescriptor(
            name=tool_name, description=tool_description, params=output_model
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_method:
                formatted_model = output_model.model_validate(args[1])
                kwargs.update(
                    {
                        name: getattr(formatted_model, name)
                        for name in params_models.keys()
                    }
                )
                self_instance = args[0]
                return func(self_instance, **kwargs)
            else:
                formatted_model = output_model.model_validate(args[0])
                kwargs.update(
                    {
                        name: getattr(formatted_model, name)
                        for name in params_models.keys()
                    }
                )

                return func(**kwargs)

        return JAImsFunctionTool(
            function=wrapper,
            descriptor=descriptor,
        )

    return decorator
