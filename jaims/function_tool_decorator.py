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
    """
    Decorator function for creating JAIms function tools.

    This is an experimental feature and I'm planning to improve it using reflection and other techniques to better infer the types of the parameters and return values.
    Right now the decorated function MUST have type annotations in order for the decorator to work, and the decorator supports only simple python types and pydantic models.

    When the decorated function expects only a pydantic model, it is forwarded as-is to the LLM (parsed with its json schema), when the function expects multiple parameters, The decorator creates a new pydantic model by inspecting the function signature.
    You can customize the name of the generated model by passing the return_value_object_name parameter (defaults to "ToolResponse")

    Args:
        name (Optional[str]): The name of the tool. If not provided, the name of the decorated function will be used.
        description (Optional[str]): The description of the tool. If not provided, an empty string will be used.
        params_descriptions (Optional[Dict[str, str]]): A dictionary mapping parameter names to their descriptions.
            If not provided, fields will have an empty string as description.
        return_value_object_name (str): The name of the return value object. Defaults to "ToolResponse". Used only when the function expects multiple parameters.

    Returns:
        The decorated function as a JAImsFunctionTool.

    Raises:
        ValueError: If a parameter has no type annotation.

    Example usage:

    ```python
        @jaimsfunctiontool(name="MyTool", description="This is my tool", params_descriptions={"param1": "The first parameter", "param2": "The second parameter"})
        def my_function(param1: int, param2: str = "default"):
            return param1 + len(param2)
    ```
    """

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
