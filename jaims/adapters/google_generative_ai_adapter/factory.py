from typing import List, Optional
from google.generativeai.types import content_types
from google.generativeai.types import generation_types

from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from .adapter import JAImsGoogleGenerativeAIAdapter


def create_jaims_gemini(
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None,
    generation_config: Optional[generation_types.GenerationConfigType] = None,
    tool_config: Optional[content_types.ToolConfigType] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
) -> JAImsAgent:
    """
    Creates a JAIms instance with a Google Cloud AI adapter.

    Args:
        project_id (str): The Google Cloud project ID.
        location (str): The Google Cloud location.
        model (JAImsGeminiModel): The Gemini model to use. Defaults to JAImsGeminiModel.GEMINI_1_5_PRO.
        generation_config (Optional[GenerationConfig]): The generation configuration. Defaults to None.
        tool_config (Optional[ToolConfig]): The tool configuration. Defaults to None.
        history_manager (Optional[JAImsHistoryManager]): The history manager. Defaults to None.
        tool_manager (Optional[JAImsToolManager]): The tool manager. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]]): The list of function tools. Defaults to None.

    Returns:
        JAImsAgent: The JAIms agent, initialized with the Google Cloud AI adapter.
    """

    adapter = JAImsGoogleGenerativeAIAdapter(
        api_key=api_key,
        model=model,
        generation_config=generation_config,
        tool_config=tool_config,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )

    return agent
