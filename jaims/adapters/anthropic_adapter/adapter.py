from __future__ import annotations
import asyncio
import json
from typing import Generator, Tuple, Union, List, Optional, Dict, Literal
from PIL import Image

from anthropic import AsyncAnthropic, Anthropic
from anthropic.types import (
    ToolParam,
    ToolUseBlock,
    TextBlock,
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
)

from anthropic.types import Message as ClaudeMessage

from anthropic.types.message_create_params import (
    ToolChoiceToolChoiceAny,
    ToolChoiceToolChoiceAuto,
    ToolChoiceToolChoiceTool,
    ToolChoice,
)

from ...interfaces import LLMAdapterITF
from ...entities import (
    ImageContent,
    Message,
    StreamingMessage,
    ToolCall,
    FunctionTool,
    MessageRole,
    Config,
)
from ..shared.image_utilities import image_to_b64

import os
from copy import deepcopy


class AnthropicParams:
    """
    Model parameters for the Anthropic API.

    Args:
        model (str, optional): The model to use. Defaults to "claude-3-opus-20240229".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
        temperature (float, optional): The sampling temperature. Defaults to 0.7.
        top_p (Optional[float], optional): The nucleus sampling parameter. Defaults to None.
        stop_sequences (Optional[List[str]], optional): A list of sequences to stop generation at. Defaults to None.
        metadata (Optional[Dict[str, str]], optional): Metadata to include with the message. Defaults to None.
        tool_choice (Optional[ToolChoice], optional): The tool choice to use. Defaults to None.

    Returns:
        AnthropicParams: The model parameters for the Anthropic API.
    """

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.metadata = metadata
        self.tool_choice = tool_choice

    def to_dict(self):
        """
        Convert the parameters to a dictionary.

        Returns:
            dict: The parameters as a dictionary.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "metadata": self.metadata,
            "tool_choice": self.tool_choice,
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def from_dict(kwargs: dict) -> AnthropicParams:
        """
        Create an instance of AnthropicParams from a dictionary.

        Args:
            kwargs (dict): The dictionary of parameters.

        Returns:
            AnthropicParams: The model parameters for the Anthropic API.
        """
        return AnthropicParams(**kwargs)

    def copy_with_overrides(self, **kwargs) -> AnthropicParams:
        """
        Create a copy of the parameters with the specified overrides.

        Args:
            **kwargs: The parameters to override.

        Returns:
            AnthropicParams: The model parameters for the Anthropic API.
        """

        new_kwargs = self.to_dict()
        new_kwargs.update(kwargs)
        return AnthropicParams.from_dict(new_kwargs)


class AnthropicAdapter(LLMAdapterITF):
    """
    Adapter for the Anthropic API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        params: Optional[Union[AnthropicParams, Dict]] = None,
        provider: Literal["anthropic", "vertex"] = "anthropic",
        existing_params_messages_behaviour: Literal["append", "replace"] = "append",
        existing_tools_behavior: Literal["append", "replace"] = "append",
    ):
        """
        Returns an instance of the AnthropicAdapter.

        Args:
            api_key (Optional[str], optional): The API key for the Anthropic API. Defaults to None.
            config (Optional[Config], optional): The configuration for the adapter. Defaults to None.
            params (Optional[Union[AnthropicParams, Dict]], optional): The model parameters for the Anthropic API. Defaults to None.
            provider (Literal["anthropic", "vertex"], optional): The provider to use. Defaults to "anthropic".
            existing_params_messages_behaviour (Literal["append", "replace"], optional): The behavior for the messages in the kwargs when receiving new messages on calls. Defaults to "append", which appends new messages to the params passed in the constructor.
            existing_tools_behaviour (Literal["append", "replace"], optional): The behavior for the tools in the kwargs when receiving new tools on calls. Defaults to "append", which appends new tools to the params passed in the constructor.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise Exception("Anthropic API key not provided.")

        self.conrig = config or Config()
        self.params = params or AnthropicParams()
        self.existing_params_messages_behaviour = existing_params_messages_behaviour
        self.existing_tools_behavior = existing_tools_behavior
        self.provider = provider

    def __get_args(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ):

        if isinstance(self.params, AnthropicParams):
            args = self.params.to_dict()
        else:
            args = deepcopy(self.params)

        sys_message, claude_messages = self.__jaims_messages_to_claude(messages or [])
        if self.existing_params_messages_behaviour == "append":
            kwargs_messages = args.get("messages", [])
            claude_messages = kwargs_messages + claude_messages
            existing_sys_message = args.get("system", None)
            if existing_sys_message:
                sys_message = (
                    existing_sys_message + sys_message
                    if sys_message
                    else existing_sys_message
                )

        args["messages"] = claude_messages

        if sys_message:
            args["system"] = sys_message

        claude_tools = self.__jaims_tools_to_claude(tools or [])

        if self.existing_tools_behavior == "append":
            claude_tools = args.get("tools", []) + claude_tools

        if len(claude_tools) > 0:
            if tool_constraints is None or len(tool_constraints) == 0:
                tool_choice = ToolChoiceToolChoiceAuto(type="auto")
            elif len(tool_constraints) == 1:
                tool_choice = ToolChoiceToolChoiceTool(
                    type="tool", name=tool_constraints[0]
                )
            else:
                tool_choice = ToolChoiceToolChoiceAny(type="any")
            args["tools"] = claude_tools
            args["tool_choice"] = tool_choice

        return args

    def call(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Message:
        args = self.__get_args(messages, tools, tool_constraints)

        if self.provider == "anthropic":
            client = Anthropic(
                api_key=self.api_key,
                max_retries=self.conrig.max_retries,
                **self.conrig.platform_specific_options,
            )
            response = client.messages.create(**args)

        elif self.provider == "vertex":
            from anthropic import AnthropicVertex

            client = AnthropicVertex(
                max_retries=self.conrig.max_retries,
                **self.conrig.platform_specific_options,
            )
            response = client.messages.create(**args)

        return self.__claude_message_to_jaims_message(response)

    def call_streaming(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[StreamingMessage, None, None]:
        args = self.__get_args(messages, tools, tool_constraints)

        async def stream_generator():

            if self.provider == "anthropic":
                client = AsyncAnthropic(
                    api_key=self.api_key,
                    max_retries=self.conrig.max_retries,
                    **self.conrig.platform_specific_options,
                )
            elif self.provider == "vertex":
                from anthropic import AsyncAnthropicVertex

                client = AsyncAnthropicVertex(
                    max_retries=self.conrig.max_retries,
                    **self.conrig.platform_specific_options,
                )

            async with client.messages.stream(**args) as stream:

                async for text in stream.text_stream:
                    snapshot = self.__claude_message_to_jaims_message(
                        stream.current_message_snapshot
                    )
                    yield StreamingMessage(
                        message=snapshot,
                        textDelta=text,
                    )

                message = await stream.get_final_message()
                final_message = self.__claude_message_to_jaims_message(message)
                yield StreamingMessage(message=final_message, textDelta=None)

        def sync_generator():
            loop = asyncio.get_event_loop()
            async_gen = stream_generator()
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break

        return sync_generator()

    def __jaims_messages_to_claude(
        self, messages: List[Message]
    ) -> Tuple[Optional[str], List[MessageParam]]:

        def jaims_role_to_claude_role(
            role: MessageRole,
        ) -> Literal["user", "assistant"]:
            if role in [MessageRole.USER, MessageRole.TOOL]:
                return "user"
            return "assistant"

        system_message = None
        claude_messages = []

        for m in messages:
            content = []
            if m.contents:
                for c in m.contents:
                    if isinstance(c, str):
                        content.append(TextBlockParam(type="text", text=c))
                    elif isinstance(c, ImageContent):
                        if isinstance(c.image, str):
                            content.append(
                                image_content=ImageBlockParam(
                                    type="image",
                                    source={
                                        "type": "base64",  # type: ignore
                                        "media_type": "image/png",
                                        "data": c.image,
                                    },
                                )
                            )
                        elif isinstance(c.image, Image.Image):
                            mime, b64 = image_to_b64(c.image)
                            content.append(
                                image_content=ImageBlockParam(
                                    type="image",
                                    source={
                                        "type": "base64",
                                        "media_type": mime,  # type: ignore
                                        "data": b64,  # type: ignore
                                    },
                                )
                            )

            if m.tool_responses:
                for tr in m.tool_responses:
                    content.append(
                        ToolResultBlockParam(
                            tool_use_id=tr.tool_call_id,
                            type="tool_result",
                            content=json.dumps(tr.response),
                            is_error=tr.is_error,
                        )
                    )

            if m.tool_calls:
                for tc in m.tool_calls:
                    content.append(
                        ToolUseBlockParam(
                            id=tc.id,
                            type="tool_use",
                            name=tc.tool_name,
                            input=tc.tool_args,
                        )
                    )

            if m.role == MessageRole.SYSTEM:
                system_message = (
                    "\n".join(c["text"] for c in content)
                    if system_message is None
                    else system_message + "\n".join(c["text"] for c in content)
                )
                continue

            claude_messages.append(
                MessageParam(
                    role=jaims_role_to_claude_role(m.role),
                    content=content,
                )
            )

        return system_message, claude_messages

    def __jaims_tools_to_claude(self, tools: List[FunctionTool]) -> List[ToolParam]:
        claude_tools = []
        for t in tools:
            claude_tool = {
                "name": t.descriptor.name,
                "description": t.descriptor.description,
                "input_schema": t.descriptor.get_json_schema(),
            }
            claude_tools.append(claude_tool)
        return claude_tools

    def __claude_message_to_jaims_message(
        self, claude_message: ClaudeMessage
    ) -> Message:

        contents = []
        tool_calls = []

        for content in claude_message.content:
            if isinstance(content, TextBlock):
                contents.append(content.text)
            elif isinstance(content, ToolUseBlock):
                tool_calls.append(
                    ToolCall(
                        id=content.id,
                        tool_name=content.name,
                        tool_args=content.input,  # type: ignore
                    )
                )

        return Message(
            role=MessageRole.ASSISTANT,
            contents=contents,
            tool_calls=tool_calls,
            raw=claude_message,
        )
