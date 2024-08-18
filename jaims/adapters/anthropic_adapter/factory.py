from typing import Dict, List, Optional, Union, Literal
from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool, JAImsOptions
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from .adapter import (
    JAImsAnthropicAdapter,
    JAImsAnthropicKWArgs,
)


def create_jaims_anthropic(
    api_key: Optional[str] = None,
    provider: Literal["anthropic", "vertex"] = "anthropic",
    options: Optional[JAImsOptions] = None,
    kwargs: Optional[Union[JAImsAnthropicKWArgs, Dict]] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
) -> JAImsAgent:

    adapter = JAImsAnthropicAdapter(
        api_key=api_key,
        provider=provider,
        options=options,
        kwargs=kwargs,
        kwargs_messages_behavior=kwargs_messages_behavior,
        kwargs_tools_behavior=kwargs_tools_behavior,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent
