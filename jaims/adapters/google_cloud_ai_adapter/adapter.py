from ...interfaces import JAImsLLMInterface
from ...entities import (
    JAImsMessage,
    JAImsFunctionTool,
    JAImsToolCall,
    JAImsMessageRole,
    JAImsMessageContent,
    JAImsContentTypes,
    JAImsStreamingMessage,
)

from typing import Iterable, List, Generator, Optional, Union
from enum import Enum

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Part,
    Image,
    Content,
    FunctionDeclaration,
    Tool,
)

from vertexai.preview.generative_models import ToolConfig, GenerationResponse
from google.protobuf import json_format


class JAImsGeminiModel(Enum):
    """
    The OPENAI Chat GPT models available.
    """

    GEMINI_1_PRO = ("gemini-1.0-pro", 12288)
    GEMINI_1_PRO_LATEST = ("gemini-1.0-pro-latest", 12288)
    GEMINI_1_PRO_001 = ("gemini-1.0-pro-001", 12288)
    GEMINI_1_PRO_VISION = ("gemini-1.0-pro-vision", 12288)
    GEMINI_1_PRO_VISION_LATEST = ("gemini-1.0-pro-vision-latest", 12288)
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", 1048576)
    GEMINI_1_5_FLASH_LATEST = ("gemini-1.5-flash-latest", 1048576)
    GEMINI_1_5_PRO = ("gemini-1.5-pro", 1048576)
    GEMINI_1_5_PRO_LATEST = ("gemini-1.5-pro-latest", 1048576)

    def __init__(self, string, max_tokens):
        self.string = string
        self.max_tokens = max_tokens

    def __str__(self):
        return self.string


class JAImsGoogleCloudAIAdapter(JAImsLLMInterface):
    def __init__(
        self,
        project_id: str,
        location: str,
        model: JAImsGeminiModel,
        generation_config: GenerationConfig,
        tool_config: ToolConfig,
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.generation_config = generation_config
        self.tool_config = tool_config
        vertexai.init(project=self.project_id, location=self.location)

    def call(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> JAImsMessage:
        response = self.__get_gemini_response(messages, tools)
        assert isinstance(response, GenerationResponse)
        return self.__vertex_to_jaims_message(response)

    def call_streaming(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> Generator[JAImsStreamingMessage, None, None]:
        response = self.__get_gemini_response(messages, tools, stream=True)
        assert isinstance(response, Iterable)
        for r in response:
            yield JAImsStreamingMessage(
                textDelta=r.text, message=self.__vertex_to_jaims_message(r)
            )

    def __jaims_role_to_vertex(self, role: JAImsMessageRole) -> str:
        if role == JAImsMessageRole.USER:
            return "user"
        else:
            return "model"

    def __jaims_tools_to_vertex(
        self, jaims_tools: List[JAImsFunctionTool]
    ) -> List[Tool]:
        function_declarations = []
        for jaims_tool in jaims_tools:
            function_declarations.append(
                FunctionDeclaration(
                    name=jaims_tool.function_tool.name,
                    description=jaims_tool.function_tool.description,
                    parameters=jaims_tool.function_tool.get_jsonapi_schema(),
                ),
            )

        return [Tool(function_declarations=function_declarations)]

    def __jaims_messages_to_vertex(
        self, messages: List[JAImsMessage]
    ) -> tuple[List[str], List[Content]]:
        contents = []
        # TODO: handle images
        system_instruction = []

        for message in messages:

            if message.role == JAImsMessageRole.SYSTEM:
                system_instruction.append(message.get_text())
                continue

            if message.role == JAImsMessageRole.USER:
                parts = []
                if message.contents:
                    for content in message.contents:
                        if content.type == JAImsContentTypes.TEXT:
                            parts.append(Part.from_text(content.content))

                        # TODO: Add image support
                contents.append(
                    Content(
                        role=self.__jaims_role_to_vertex(message.role),
                        parts=parts,
                    )
                )
                continue

            if message.tool_response:
                contents.append(
                    Content(
                        role=self.__jaims_role_to_vertex(message.role),
                        parts=[
                            Part.from_function_response(
                                name=message.tool_response.tool_name,
                                response=message.tool_response.response,
                            )
                        ],
                    )
                )
                continue

            if message.tool_calls:
                raw_message = message.raw
                assert isinstance(raw_message, GenerationResponse)
                contents.extend(raw_message.candidates[0].function_calls)
                continue

        return system_instruction, contents

    def __vertex_to_jaims_message(self, response: GenerationResponse) -> JAImsMessage:
        assert response.candidates is not None

        candidate = response.candidates[0]
        role = JAImsMessageRole.ASSISTANT
        tool_calls = None
        contents = None
        if candidate.function_calls:
            role = JAImsMessageRole.TOOL
            tool_calls = [
                JAImsToolCall(
                    id=call.name,
                    tool_name=call.name,
                    tool_args=json_format.MessageToDict(call.args),
                )
                for call in candidate.function_calls
            ]

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    contents = [
                        JAImsMessageContent(
                            type=JAImsContentTypes.TEXT,
                            content=part.text,
                        )
                    ]
                    # TODO: Add image support

        return JAImsMessage(
            role=role,
            contents=contents,
            tool_calls=tool_calls,
            raw=response,
        )

    def __get_gemini_response(
        self,
        messages: List[JAImsMessage],
        tools: List[JAImsFunctionTool],
        stream: bool = False,
    ):

        # TODO: Add error and rate limit handling

        system_instruction, vertex_messages = self.__jaims_messages_to_vertex(messages)

        multimodal_model = GenerativeModel(
            self.model.value,
            generation_config=self.generation_config.to_dict(),
            tools=self.__jaims_tools_to_vertex(tools),
            tool_config=self.tool_config,
            system_instruction=system_instruction,  # type: ignore
        )

        response = multimodal_model.generate_content(
            contents=vertex_messages,
            stream=stream,
        )

        return response
