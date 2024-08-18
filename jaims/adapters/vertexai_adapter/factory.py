from typing import Any, Dict, List, Optional, Union, Literal
from ...agent import JAImsAgent
from ...entities import JAImsFunctionTool, JAImsOptions
from ...interfaces import JAImsHistoryManager, JAImsToolManager
from vertexai.generative_models import GenerationConfig, SafetySetting
from .adapter import JAImsVertexAIAdapter


def create_jaims_vertex(
    project_id: str,
    location: str,
    model_name: str,
    generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
    safety_settings: Optional[List[SafetySetting]] = None,
    options: Optional[JAImsOptions] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
) -> JAImsAgent:

    adapter = JAImsVertexAIAdapter(
        project_id=project_id,
        location=location,
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
        options=options,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent
