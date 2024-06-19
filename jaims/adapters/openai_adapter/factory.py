from typing import Dict, List, Optional, Union
from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool, JAImsOptions, JAImsLLMConfig
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from .adapter import (
    JAImsOpenaiAdapter,
    JAImsOpenaiKWArgs,
    OpenAITransactionStorageInterface,
)


def create_jaims_openai(
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    kwargs: Optional[Union[JAImsOpenaiKWArgs, Dict]] = None,
    transaction_storage: Optional[OpenAITransactionStorageInterface] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
) -> JAImsAgent:

    adapter = JAImsOpenaiAdapter(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        transaction_storage=transaction_storage,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )

    return agent
