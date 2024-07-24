from typing import Any, Dict, List, Literal, Optional, Union
from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool, JAImsOptions
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from .adapter import (
    JAImsMistralAdapter,
    JAImsMistralKWArgs,
    MistralTransactionStorageInterface,
)


def create_jaims_mistral(
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    kwargs: Optional[Union[JAImsMistralKWArgs, Dict[str, Any]]] = None,
    transaction_storage: Optional[MistralTransactionStorageInterface] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
) -> JAImsAgent:
    adapter = JAImsMistralAdapter(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        transaction_storage=transaction_storage,
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
