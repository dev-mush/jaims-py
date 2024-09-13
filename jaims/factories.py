from __future__ import annotations
from typing import Literal, Optional, List, TYPE_CHECKING


from .agent import Agent

from jaims.entities import Config, LLMParams
from jaims.interfaces import HistoryManagerITF, ToolManagerITF, FunctionTool


def openai_factory(
    model: str,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
    llm_params: Optional[LLMParams] = None,
    history_manager: Optional[HistoryManagerITF] = None,
    tool_manager: Optional[ToolManagerITF] = None,
    tools: Optional[List[FunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> Agent:
    """
    Factory function to create an instance of JAIms Agent using OpenAI as the underlying model provider.

    Args:
        model (str): The name or identifier of the OpenAI model to use.
        api_key (Optional[str]): The API key for accessing the OpenAI service. Defaults to None.
        config (Optional[Config]): Additional options for configuring the JAImsAgent. Defaults to None.
        llm_params (Optional[LLMParams]): Parameters for the language model. Defaults to None.
        history_manager (Optional[HistoryManager]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[ToolManager]): The tool manager to use for managing Function Calls. Defaults to None.
        tools (Optional[List[FunctionTool]]): The list of Function Tools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): The list of tool constraints to use. Defaults to None.

    Returns:
        Agent: An instance of JAIms Agent configured with the OpenAI model.

    """
    from .adapters.openai_adapter import JAImsOpenaiKWArgs, JAImsOpenaiAdapter

    llm_params = llm_params or LLMParams()
    config = config or Config()
    tool_choice = None

    kwargs = JAImsOpenaiKWArgs().copy_with_overrides(
        model=model,
        temperature=llm_params.temperature,
        max_tokens=llm_params.max_tokens,
        response_format=llm_params.response_format,
        tool_choice=tool_choice,
    )

    adapter = JAImsOpenaiAdapter(
        api_key=api_key,
        options=config,
        kwargs=kwargs,
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent


def google_factory(
    model: str,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
    llm_params: Optional[LLMParams] = None,
    history_manager: Optional[HistoryManagerITF] = None,
    tool_manager: Optional[ToolManagerITF] = None,
    tools: Optional[List[FunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> Agent:
    """
    Factory function to create an instance of JAImsAgent using Google Generative AI as the underlying model.

    Args:
        model (str): The name or identifier of the Google model to use.
        api_key (Optional[str]): The API key for accessing the Google model. Defaults to None.
        config (Optional[Config]): Additional options for configuring the JAImsAgent. Defaults to None.
        llm_params (Optional[LLMParams]): Parameters for the language model. Defaults to None.
        history_manager (Optional[HistoryManagerITF]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[ToolManagerITF]): The tool manager to use for managing Function Calls. Defaults to None.
        tools (Optional[List[FunctionTool]]): The list of Function Tools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): The list of tool constraints to use. Defaults to None.

    Returns:
        Agent: An instance of JAIms Agent configured with the Google model.
    """

    from .adapters.google_generative_ai_adapter import JAImsGoogleGenerativeAIAdapter
    from google.generativeai.types import generation_types

    llm_params = llm_params or LLMParams()
    config = config or Config()

    generation_config = generation_types.GenerationConfig(
        temperature=llm_params.temperature,
        max_output_tokens=llm_params.max_tokens,
        response_schema=llm_params.response_format,
    )

    adapter = JAImsGoogleGenerativeAIAdapter(
        api_key=api_key,
        model=model,
        generation_config=generation_config,
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent


def mistral_factory(
    model: str,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
    llm_params: Optional[LLMParams] = None,
    history_manager: Optional[HistoryManagerITF] = None,
    tool_manager: Optional[ToolManagerITF] = None,
    tools: Optional[List[FunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> Agent:
    """
    Factory function to create an instance of JAImsAgent using Mistral as the underlying model.

    Args:
        model (str): The name or identifier of the Mistral model to use.
        api_key (Optional[str]): The API key for accessing the Mistral service. Defaults to None.
        config (Optional[Config]): Additional options for configuring the JAIms Agent. Defaults to None.
        llm_params (Optional[LLMParams]): Parameters for the language model. Defaults to None.
        history_manager (Optional[HistoryManagerITF]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[ToolManagerITF]): The tool manager to use for managing Function Calls. Defaults to None.
        tools (Optional[List[FunctionTool]]): The list of Function Tools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): The list of tool constraints to use. Defaults to None.

    Returns:
        Agent: An instance of JAIms Agent configured with the Mistral model.

    """
    from .adapters.mistral_adapter import JAImsMistralKWArgs, JAImsMistralAdapter

    llm_params = llm_params or LLMParams()
    config = config or Config()
    tool_choice = None

    kwargs = JAImsMistralKWArgs().copy_with_overrides(
        model=model,
        temperature=llm_params.temperature,
        max_tokens=llm_params.max_tokens,
        response_format=llm_params.response_format,
        tool_choice=tool_choice,
    )

    adapter = JAImsMistralAdapter(
        api_key=api_key,
        options=config,
        kwargs=kwargs,
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent


def anthropic_factory(
    model: str,
    api_key: Optional[str] = None,
    provider: Literal[
        "anthropic",
        "vertex",
    ] = "anthropic",
    config: Optional[Config] = None,
    llm_params: Optional[LLMParams] = None,
    history_manager: Optional[HistoryManagerITF] = None,
    tool_manager: Optional[ToolManagerITF] = None,
    tools: Optional[List[FunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> Agent:
    """
    Factory function to create an instance of JAImsAgent using Anthropic as the underlying model.

    Args:
        model (str): The name or identifier of the Anthropic model to use.
        api_key (Optional[str]): The API key for accessing the Anthropic service. Defaults to None.
        provider (Literal["anthropic", "vertex"]): The provider of the Anthropic model. Defaults to "anthropic".
        config (Optional[Config]): Additional options for configuring the JAIms Agent. Defaults to None.
        llm_params (Optional[LLMParams]): Parameters for the language model. Defaults to None.
        history_manager (Optional[HistoryManagerITF]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[ToolManagerITF]): The tool manager to use for managing Function Calls. Defaults to None.
        tools (Optional[List[FunctionTool]]): The list of Function Tools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): The list of tool constraints to use. Defaults to None.

    Returns:
        Agent: An instance of JAIms Agent configured with the Anthropic model.
    """
    from .adapters.anthropic_adapter import JAImsAnthropicKWArgs, JAImsAnthropicAdapter

    llm_params = llm_params or LLMParams()
    config = config or Config()

    kwargs = JAImsAnthropicKWArgs().copy_with_overrides(
        model=model,
        temperature=llm_params.temperature,
        max_tokens=llm_params.max_tokens,
    )

    adapter = JAImsAnthropicAdapter(
        api_key=api_key,
        provider=provider,
        options=config,
        kwargs=kwargs,
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent


def vertex_ai_factory(
    model: str,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
    llm_params: Optional[LLMParams] = None,
    history_manager: Optional[HistoryManagerITF] = None,
    tool_manager: Optional[ToolManagerITF] = None,
    tools: Optional[List[FunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> Agent:
    """
    Factory function to create an instance of JAImsAgent using Vertex AI as the underlying model.

    Args:
        model (str): The name or identifier of the Vertex AI model to use.
        api_key (Optional[str]): The API key for accessing the Vertex AI service. Defaults to None.
        config (Optional[Config]): Additional options for configuring the JAIms Agent. Defaults to None.
        llm_params (Optional[LLMParams]): Parameters for the language model. Defaults to None.
        history_manager (Optional[HistoryManagerITF]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[ToolManagerITF]): The tool manager to use for managing Function Calls. Defaults to None.
        tools (Optional[List[FunctionTool]]): The list of Function Tools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): The list of tool constraints to use. Defaults to None.

    Returns:
        Agent: An instance of JAIms Agent configured with the Vertex AI model.
    """

    from .adapters.vertexai_adapter import JAImsVertexAIAdapter
    from vertexai.generative_models import GenerationConfig

    if not config:
        raise ValueError("pass project_id and location in options")

    if config.platform_specific_options["project_id"] is None:
        raise ValueError("project_id is required in options")

    if config.platform_specific_options["location"] is None:
        raise ValueError("location is required in options")

    project_id = config.platform_specific_options["project_id"]
    location = config.platform_specific_options["location"]

    llm_params = llm_params or LLMParams()

    generation_config = GenerationConfig(
        temperature=llm_params.temperature,
        max_output_tokens=llm_params.max_tokens,
        response_schema=llm_params.response_format,
    )

    adapter = JAImsVertexAIAdapter(
        project_id=project_id,
        location=location,
        model_name=model,
        generation_config=generation_config,
        options=config,
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )

    return agent
