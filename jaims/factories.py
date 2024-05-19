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
) -> JAImsAgent:

    from .adapters.openai_adapter import create_jaims_openai
    from .adapters.openai_adapter import JAImsOpenaiKWArgs

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    kwargs = JAImsOpenaiKWArgs().copy_with_overrides(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format=config.response_format,
    )

    return create_jaims_openai(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )


def google_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
) -> JAImsAgent:

    from .adapters.google_generative_ai_adapter.factory import (
        create_jaims_gemini,
        generation_types,
    )

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    generation_config = generation_types.GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        response_schema=config.response_format,
    )

    return create_jaims_gemini(
        api_key=api_key,
        model=model,
        generation_config=generation_config,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )
