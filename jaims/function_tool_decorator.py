from jaims.entities import (
    JAImsFunctionTool,
    JAImsFunctionToolDescriptor,
)
from typing import Any, Optional, Dict
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
        has_base_model = False

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
                # check if the param is a BaseModel or a subclass of it
                if isinstance(param.annotation, type) and issubclass(
                    param.annotation, BaseModel
                ):
                    has_base_model = True

                params_models[param_name] = (
                    param.annotation,
                    Field(default=param.default, description=param_description),
                )

        if len(params_models) == 1 and has_base_model:
            output_model = list(params_models.values())[0][0]
        else:
            output_model = create_model(return_value_object_name, **params_models)

        descriptor = JAImsFunctionToolDescriptor(
            name=tool_name, description=tool_description, params=output_model
        )

        def formatter(data: Dict[str, Any]):
            formatted_model = output_model.model_validate(data)
            kwargs = {
                name: getattr(formatted_model, name) for name in params_models.keys()
            }

            return (), kwargs

        return JAImsFunctionTool(
            function=func,
            formatter=formatter,
            descriptor=descriptor,
        )

    return decorator
