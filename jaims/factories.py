from __future__ import annotations
from typing import Literal, Optional, List, TYPE_CHECKING


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
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
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
    from .adapters.openai_adapter import JAImsOpenaiKWArgs, JAImsOpenaiAdapter

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

    adapter = JAImsOpenaiAdapter(
        api_key=api_key,
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

    from .adapters.google_generative_ai_adapter import JAImsGoogleGenerativeAIAdapter
    from google.generativeai.types import generation_types

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    generation_config = generation_types.GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        response_schema=config.response_format,
    )

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
        tool_constraints=tool_constraints,
    )

    return agent


def mistral_factory(
    model: str,
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    config: Optional[JAImsLLMConfig] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
    tool_constraints: Optional[List[str]] = None,
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
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
    from .adapters.mistral_adapter import JAImsMistralKWArgs, JAImsMistralAdapter

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

    adapter = JAImsMistralAdapter(
        api_key=api_key,
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
    kwargs_messages_behavior: Literal["append", "replace"] = "append",
    kwargs_tools_behavior: Literal["append", "replace"] = "append",
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
    from .adapters.anthropic_adapter import JAImsAnthropicKWArgs, JAImsAnthropicAdapter

    config = config or JAImsLLMConfig()
    options = options or JAImsOptions()

    kwargs = JAImsAnthropicKWArgs().copy_with_overrides(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

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

    from .adapters.vertexai_adapter import JAImsVertexAIAdapter
    from vertexai.generative_models import GenerationConfig

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

    adapter = JAImsVertexAIAdapter(
        project_id=project_id,
        location=location,
        model_name=model,
        generation_config=generation_config,
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
