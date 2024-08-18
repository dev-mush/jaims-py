from __future__ import annotations
from typing import Literal, Optional, List, TYPE_CHECKING

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
    """
    Factory function to create an instance of JAImsAgent using OpenAI as the underlying model.

    Args:
        model (str): The name or identifier of the OpenAI model to use.
        api_key (Optional[str]): The API key for accessing the OpenAI service. Defaults to None.
        options (Optional[JAImsOptions]): Additional options for configuring the JAImsAgent. Defaults to None.
        config (Optional[JAImsLLMConfig]): Configuration options specific to the OpenAI language model. Defaults to None.
        history_manager (Optional[JAImsHistoryManager]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[JAImsToolManager]): The tool manager to use for managing JAImsFunctionTools. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]]): The list of JAImsFunctionTools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): An optional list of tool constraints. Defaults to None.

    Returns:
        JAImsAgent: An instance of JAImsAgent configured with the OpenAI model.

    """
    from .adapters.openai_adapter import create_jaims_openai
    from .adapters.openai_adapter import JAImsOpenaiKWArgs

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()
    tool_choice = None

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
    """
    Factory function to create an instance of JAImsAgent using Google Gemini as the underlying model.

    Args:
        model (str): The name or ID of the Google model to use.
        api_key (Optional[str]): The API key for accessing the Google model (default: None).
        options (Optional[JAImsOptions]): Additional options for the JAImsAgent (default: None).
        config (Optional[JAImsLLMConfig]): Configuration for the JAImsLLM model (default: None).
        history_manager (Optional[JAImsHistoryManager]): History manager for the JAImsAgent (default: None).
        tool_manager (Optional[JAImsToolManager]): Tool manager for the JAImsAgent (default: None).
        tools (Optional[List[JAImsFunctionTool]]): List of function tools for the JAImsAgent (default: None).
        tool_constraints (Optional[List[str]]): List of tool constraints for the JAImsAgent (default: None).

    Returns:
        JAImsAgent: An instance of JAImsAgent configured for Google models.
    """

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
        tool_constraints=tool_constraints,
    )


def mistral_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> JAImsAgent:
    """
    Factory function to create an instance of JAImsAgent using Mistral as the underlying model.

    Args:
        model (str): The name or identifier of the OpenAI model to use.
        api_key (Optional[str]): The API key for accessing the OpenAI service. Defaults to None.
        options (Optional[JAImsOptions]): Additional options for configuring the JAImsAgent. Defaults to None.
        config (Optional[JAImsLLMConfig]): Configuration options specific to the OpenAI language model. Defaults to None.
        history_manager (Optional[JAImsHistoryManager]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[JAImsToolManager]): The tool manager to use for managing JAImsFunctionTools. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]]): The list of JAImsFunctionTools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): An optional list of tool constraints. Defaults to None.

    Returns:
        JAImsAgent: An instance of JAImsAgent configured with the OpenAI model.

    """
    from .adapters.mistral_adapter import create_jaims_mistral
    from .adapters.mistral_adapter import JAImsMistralKWArgs

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()
    tool_choice = None

    kwargs = JAImsMistralKWArgs().copy_with_overrides(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format=config.response_format,
        tool_choice=tool_choice,
    )

    return create_jaims_mistral(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )


def anthropic_factory(
    model: str,
    api_key: Optional[str] = None,
    provider: Literal[
        "anthropic",
        "vertex",
    ] = "anthropic",
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> JAImsAgent:
    """
    Factory function to create an instance of JAImsAgent using Anthropic as the underlying model.

    Args:
        model (str): The name or identifier of the OpenAI model to use.
        api_key (Optional[str]): The API key for accessing the OpenAI service. Defaults to None.
        provider (Literal["anthropic", "vertex"]): The provider to use for the Anthropic model. Defaults to "anthropic".
        options (Optional[JAImsOptions]): Additional options for configuring the JAImsAgent. Defaults to None.
        config (Optional[JAImsLLMConfig]): Configuration options specific to the OpenAI language model. Defaults to None.
        history_manager (Optional[JAImsHistoryManager]): The history manager to use for managing conversation history. Defaults to None.
        tool_manager (Optional[JAImsToolManager]): The tool manager to use for managing JAImsFunctionTools. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]]): The list of JAImsFunctionTools to use. Defaults to None.
        tool_constraints (Optional[List[str]]): An optional list of tool constraints. Defaults to None.

    Returns:
        JAImsAgent: An instance of JAImsAgent configured with the OpenAI model.

    """
    from .adapters.anthropic_adapter import create_jaims_anthropic
    from .adapters.anthropic_adapter import JAImsAnthropicKWArgs

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    kwargs = JAImsAnthropicKWArgs().copy_with_overrides(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return create_jaims_anthropic(
        api_key=api_key,
        options=options,
        provider=provider,
        kwargs=kwargs,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )


def vertex_ai_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
) -> JAImsAgent:
    """
    Factory function to create an instance of JAImsAgent using Google Gemini as the underlying model.

    Args:
        model (str): The name or ID of the Google model to use.
        api_key (Optional[str]): The API key for accessing the Google model (default: None).
        options (Optional[JAImsOptions]): Additional options for the JAImsAgent (default: None).
        config (Optional[JAImsLLMConfig]): Configuration for the JAImsLLM model (default: None).
        history_manager (Optional[JAImsHistoryManager]): History manager for the JAImsAgent (default: None).
        tool_manager (Optional[JAImsToolManager]): Tool manager for the JAImsAgent (default: None).
        tools (Optional[List[JAImsFunctionTool]]): List of function tools for the JAImsAgent (default: None).
        tool_constraints (Optional[List[str]]): List of tool constraints for the JAImsAgent (default: None).

    Returns:
        JAImsAgent: An instance of JAImsAgent configured for Google models.
    """

    from .adapters.vertexai_adapter.factory import (
        create_jaims_vertex,
        GenerationConfig,
    )

    if not options:
        raise ValueError("pass project_id and location in options")

    if options.platform_specific_options["project_id"] is None:
        raise ValueError("project_id is required in options")

    if options.platform_specific_options["location"] is None:
        raise ValueError("location is required in options")

    project_id = options.platform_specific_options["project_id"]
    location = options.platform_specific_options["location"]

    config = config or JAImsLLMConfig()

    generation_config = GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        response_schema=config.response_format,
    )

    return create_jaims_vertex(
        model_name=model,
        project_id=project_id,
        location=location,
        generation_config=generation_config,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
        tool_constraints=tool_constraints,
    )
