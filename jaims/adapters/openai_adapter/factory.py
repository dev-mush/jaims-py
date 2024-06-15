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
    """
    Creates a JAIms instance with an OpenAI adapter.

    Args:
        api_key (Optional[str], optional): The OpenAI API key. Defaults to None.
        options (Optional[JAImsOptions], optional): The options for the adapter. Defaults to None.
        kwargs (Optional[JAImsOpenaiKWArgs], optional): The keyword arguments for the adapter. Defaults to None.
        transaction_storage (Optional[JAImsTransactionStorageInterface], optional): The transaction storage interface. Defaults to None.
        history_manager (Optional[JAImsHistoryManager], optional): The history manager. Defaults to None.
        tool_manager (Optional[JAImsToolManager], optional): The tool manager. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]], optional): The list of function tools. Defaults to None.

    Returns:
        JAImsAgent: The JAIms agent, initialized with the OpenAI adapter.
    """
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
