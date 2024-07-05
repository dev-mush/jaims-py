from __future__ import annotations
import json
from typing import Generator, Union
import anthropic
from typing import List, Optional, Dict, Literal
from PIL import Image

from ...interfaces import (
    JAImsLLMInterface,
)
from ...entities import (
    JAImsImageContent,
    JAImsContentType,
    JAImsMessage,
    JAImsStreamingMessage,
    JAImsToolCall,
    JAImsFunctionTool,
    JAImsMessageRole,
    JAImsOptions,
)
from ..shared.image_utilities import image_to_b64
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

import os
from copy import deepcopy

# ------------------------
# anthropic / LLM modeling
# ------------------------


class JAImsAnthropicAdapter(JAImsLLMInterface):

    def __init__(
        self,
        api_key: Optional[str] = None,
        options: Optional[JAImsOptions] = None,
        params: Optional[
            anthropic.types.message_create_params.MessageCreateParams
        ] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise Exception("Anthropic API key not provided.")

        self.options = options or JAImsOptions()
        self.params = params

    def __get_default_params(self, streaming: bool):

        max_tokens = 1024
        model = "claude-3-5-sonnet-20240620"

        if streaming:
            return anthropic.types.message_create_params.MessageCreateParamsStreaming(
                messages=[],
                max_tokens=1234,
                model="claude-3-5-sonnet-20240620",
                stream=True,
            )
        return anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            messages=[],
            max_tokens=1234,
            model="claude-3-opus-20240229",
            stream=False,
        )

    def __jaims_messages_to_anthropic(
        self, messages: List[JAImsMessage]
    ) -> List[anthropic.types.MessageParam]:

        def format_contents(contents: List[JAImsContentType]):
            if len(contents) == 1 and isinstance(contents[0], str):
                return contents[0]
            else:
                raw_contents = []
                for c in contents:
                    if isinstance(c, str):
                        raw_contents.append({"type": "text", "text": c})
                    elif isinstance(c, JAImsImageContent):
                        url = c.image
                        if isinstance(c.image, Image.Image):
                            mime, b64 = image_to_b64(c.image)
                            url = f"data:{mime};base64,{b64}"

                        raw_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            }
                        )

                    else:
                        raise Exception(f"Unsupported content type: {type(c)}")

                return raw_contents

        def format_tool_calls(tool_calls: List[JAImsToolCall]):
            raw_tool_calls = []
            for tc in tool_calls:
                raw_tool_call = {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.tool_args),
                    },
                }
                raw_tool_calls.append(raw_tool_call)
            return raw_tool_calls

        anthropic_messages: List[anthropic.types.MessageParam] = []
        for m in messages:
            if m.tool_response:
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="assistant",
                        content=[
                            anthropic.types.ToolResultBlockParam(
                                tool_use_id=m.tool_response.tool_call_id,
                                type="tool_result",
                                content=[],
                                is_error=False,
                            ),
                        ],
                    )
                )
                continue

            if m.tool_calls:
                raw_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": format_tool_calls(m.tool_calls),
                    }
                )
                continue

            raw_message = {}
            raw_message["role"] = m.role.value

            if m.contents:
                raw_message["content"] = format_contents(m.contents)

            raw_messages.append(raw_message)

        return raw_messages

    def __get_params(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
        stream: bool = False,
    ):

        if self.params:
            self.params["max_tokens"] = 1000

        if stream:
            return anthropic.types.message_create_params.MessageCreateParamsStreaming(
                messages=[],
                max_tokens=1234,
                model="claude-3-opus-20240229",
                stream=True,
            )
        return anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            messages=[],
            max_tokens=1234,
            model="claude-3-opus-20240229",
            stream=False,
        )

    def call(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> JAImsMessage:
        params = self.__get_params(messages, tools, tool_constraints)
        response = self.___get_openai_response(args, self.options)
        assert isinstance(response, ChatCompletion)

        return self.__openai_chat_completion_to_jaims_message(response)

    def call_streaming(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[JAImsStreamingMessage, None, None]:
        args = self.__get_args(
            messages, tools, stream=True, tool_constraints=tool_constraints
        )
        response = self.___get_openai_response(args, self.options)
        assert isinstance(response, Stream)

        accumulated_delta = None
        for completion_chunk in response:
            accumulated_delta = self.__accumulate_choice_delta(
                accumulated_delta, completion_chunk.choices[0].delta
            )
            yield self.__openai_chat_completion_choice_delta_to_jaims_message(
                accumulated_delta, completion_chunk
            )

        if self.transaction_storage and accumulated_delta:
            self.transaction_storage.store_transaction(
                request=args,
                response=accumulated_delta.model_dump(exclude_none=True),
            )
