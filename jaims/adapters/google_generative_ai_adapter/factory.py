from typing import List, Optional
from google.generativeai.types import generation_types

from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from .adapter import JAImsGoogleGenerativeAIAdapter


def create_jaims_gemini(
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None,
    generation_config: Optional[generation_types.GenerationConfigType] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
) -> JAImsAgent:

    adapter = JAImsGoogleGenerativeAIAdapter(
        api_key=api_key,
        model=model,
        generation_config=generation_config,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )

    return agent
