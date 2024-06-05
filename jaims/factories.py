from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import JAImsAgent

from jaims.entities import JAImsOptions, JAImsLLMConfig
from jaims.interfaces import JAImsHistoryManager, JAImsToolManager, JAImsFunctionTool


def openai_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> JAImsAgent:

    from .adapters.openai_adapter import create_jaims_openai
    from .adapters.openai_adapter import JAImsOpenaiKWArgs

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()
    tool_choice = None
    if tool_constraints and tools:

        if len(tool_constraints) > 1:
            raise ValueError(
                "Only one tool choice is allowed when using the OpenAI API."
            )

        tool_choice = {
            "type": "function",
            "function": {
                "name": tool_constraints[0],
            },
        }

    kwargs = JAImsOpenaiKWArgs().copy_with_overrides(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format=config.response_format,
        tool_choice=tool_choice,
    )

    return create_jaims_openai(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )


def google_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> JAImsAgent:

    from .adapters.google_generative_ai_adapter.factory import (
        create_jaims_gemini,
        generation_types,
        content_types,
    )

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    tool_config = None
    if tool_constraints and tools:
        tool_config = content_types.to_tool_config(
            {
                "function_calling_config": {
                    "mode": "any",
                    "allowed_function_names": tool_constraints,
                }
            }  # type: ignore
        )

    generation_config = generation_types.GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        response_schema=config.response_format,
    )

    return create_jaims_gemini(
        api_key=api_key,
        model=model,
        generation_config=generation_config,
        tool_config=tool_config,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )
