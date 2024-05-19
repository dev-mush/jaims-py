import os
from ...interfaces import JAImsLLMInterface, JAImsHistoryManager, JAImsToolManager
from ...entities import (
    JAImsMessage,
    JAImsFunctionTool,
    JAImsToolCall,
    JAImsMessageRole,
    JAImsImageContent,
    JAImsStreamingMessage,
    JAImsOptions,
)
from ...agent import JAImsAgent
from ..shared.image_utilities import image_to_bytes
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

from typing import Iterable, List, Generator, Optional
from PIL import Image


from google.generativeai import GenerativeModel
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
import google.ai.generativelanguage as glm


class JAImsGoogleGenerativeAIAdapter(JAImsLLMInterface):
    def __init__(
        self,
        model: str,
        tool_config: Optional[content_types.ToolConfigType] = None,
        generation_config: Optional[generation_types.GenerationConfigType] = None,
        options: Optional[JAImsOptions] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise Exception("GOOGLE_API_KEY not provided.")

        self.model = model
        self.generation_config = generation_config
        self.tool_config = tool_config
        self.options = options or JAImsOptions()

    def call(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> JAImsMessage:
        response = self.__get_gemini_response(messages, tools)
        assert isinstance(response, generation_types.GenerateContentResponse)
        return self.__gemini_to_jaims_message(response)

    def call_streaming(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> Generator[JAImsStreamingMessage, None, None]:
        response = self.__get_gemini_response(messages, tools, stream=True)
        assert isinstance(response, Iterable)
        for r in response:
            if r.candidates:
                message = self.__gemini_to_jaims_message(r)
                yield JAImsStreamingMessage(
                    textDelta=message.get_text() or None,
                    message=message,
                )

    def __jaims_role_to_gemini(self, role: JAImsMessageRole) -> str:
        if role == JAImsMessageRole.USER:
            return "user"
        else:
            return "model"

    def __jaims_tools_to_gemini(
        self, jaims_tools: List[JAImsFunctionTool]
    ) -> List[content_types.Tool]:
        function_declarations = []
        for jaims_tool in jaims_tools:
            function_declarations.append(
                content_types.FunctionDeclaration(
                    name=jaims_tool.function_tool.name,
                    description=jaims_tool.function_tool.description,
                    parameters=jaims_tool.function_tool.get_jsonapi_schema(),
                ),
            )

        return [content_types.Tool(function_declarations=function_declarations)]

    def __jaims_messages_to_gemini(
        self, jaims_messages: List[JAImsMessage]
    ) -> tuple[List[str], List[content_types.ContentsType]]:
        gemini_messages = []
        system_instruction = []

        for jaims_message in jaims_messages:
            if jaims_message.role == JAImsMessageRole.SYSTEM:
                system_instruction.append(jaims_message.get_text())
                continue

            role = self.__jaims_role_to_gemini(jaims_message.role)
            parts = []

            if jaims_message.contents:
                for content in jaims_message.contents:
                    if isinstance(content, str):
                        parts.append(glm.Part(text=content))
                    if isinstance(content, JAImsImageContent):
                        if isinstance(content.image, str):
                            raise ValueError("Image URL not supported")
                        elif isinstance(content.image, Image.Image):
                            mime, data = image_to_bytes(content.image)
                            parts.append(
                                glm.Part(
                                    inline_data=glm.Blob(mime_type=mime, data=data)
                                )
                            )
                        else:
                            raise ValueError("Invalid image content type")

            if jaims_message.tool_response:
                parts.append(
                    glm.Part(
                        function_response=glm.FunctionResponse(
                            name=jaims_message.tool_response.tool_name,
                            response={"result": jaims_message.tool_response.response},
                        )
                    )
                )

            if jaims_message.tool_calls:
                for tool_call in jaims_message.tool_calls:
                    parts.append(
                        glm.Part(
                            function_call=glm.FunctionCall(
                                name=tool_call.tool_name,
                                args=tool_call.tool_args,
                            )
                        )
                    )

            gemini_messages.append(
                glm.Content(
                    role=role,
                    parts=parts,
                )
            )

        return system_instruction, gemini_messages

    def __gemini_to_jaims_message(
        self, response: generation_types.GenerateContentResponse
    ) -> JAImsMessage:
        assert response.candidates is not None

        role = JAImsMessageRole.ASSISTANT
        tool_calls = []
        contents = []
        for part in response.parts:
            if fn := part.function_call:
                args = {}
                for k, v in fn.args.items():
                    args[k] = v
                tool_calls.append(
                    JAImsToolCall(
                        id=fn.name,
                        tool_name=fn.name,
                        tool_args=args,
                    )
                )

            if txt := part.text:
                contents.append(txt)

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

        def handle_gemini_error(error):

            # From: https://ai.google.dev/gemini-api/docs/troubleshooting
            # 400	INVALID_ARGUMENT	The request body is malformed.	Check the API reference for request format, examples, and supported versions. Using features from a newer API version with an older endpoint can cause errors.
            # 403	PERMISSION_DENIED	Your API key doesn't have the required permissions.	Check that your API key is set and has the right access.
            # 404	NOT_FOUND	The requested resource wasn't found.	Check if all parameters in your request are valid for your API version.
            # 429	RESOURCE_EXHAUSTED	You've exceeded the rate limit.	Ensure you're within the model's rate limit. Request a quota increase if needed.
            # 500	INTERNAL	An unexpected error occurred on Google's side.	Wait a bit and retry your request. If the issue persists after retrying, please report it using the Send feedback button in Google AI Studio.
            # 503	UNAVAILABLE	The service may be temporarily overloaded or down.	Wait a bit and retry your request. If the issue persists after retrying, please report it using the Send feedback button in Google AI Studio.

            if error.code == 429 or error.code == 503 or error.code == 500:
                return ErrorHandlingMethod.EXPONENTIAL_BACKOFF

            return ErrorHandlingMethod.FAIL

        def call_gemini():
            system_instruction, gemini_messages = self.__jaims_messages_to_gemini(
                messages
            )
            gemini_tools = self.__jaims_tools_to_gemini(tools) if tools else None

            multimodal_model = GenerativeModel(
                self.model,
                generation_config=self.generation_config,
                tools=gemini_tools,
                tool_config=self.tool_config,
                system_instruction=system_instruction or None,
            )

            return multimodal_model.generate_content(
                contents=gemini_messages,
                stream=stream,
            )

        return call_with_exponential_backoff(
            call_gemini, handle_gemini_error, self.options
        )
