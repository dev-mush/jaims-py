from ...interfaces import JAImsLLMInterface
from ...entities import (
    JAImsMessage,
    JAImsFunctionTool,
    JAImsToolCall,
    JAImsMessageRole,
    JAImsImageContent,
    JAImsStreamingMessage,
    JAImsOptions,
)
from ..shared.image_utilities import image_to_bytes
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

from typing import Any, Dict, Iterable, List, Generator, Optional, Tuple, Union
from PIL import Image

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    SafetySetting,
    Tool,
    ToolConfig,
    Part,
    FunctionDeclaration,
    Content,
    GenerationResponse,
)
from vertexai.generative_models import Image as VertexImage

from vertexai.generative_models._generative_models import ContentsType


class JAImsVertexAIAdapter(JAImsLLMInterface):
    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        options: Optional[JAImsOptions] = None,
    ):
        vertexai.init(project=project_id, location=location)

        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.safety_settings = safety_settings
        self.generation_config = generation_config
        self.options = options or JAImsOptions()

    def call(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> JAImsMessage:
        response = self.__get_gemini_response(messages, tools, tool_constraints)
        _, message = self.__gemini_to_jaims_message(response)
        return message

    def call_streaming(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[JAImsStreamingMessage, None, None]:
        response = self.__get_gemini_response(
            messages, tools, stream=True, tool_constraints=tool_constraints
        )
        assert isinstance(response, Iterable)
        existing_tool_calls = None
        existing_text = None
        for r in response:
            if r.candidates:
                text_delta, message = self.__gemini_to_jaims_message(
                    r, existing_text, existing_tool_calls
                )
                if message.tool_calls:
                    existing_tool_calls = message.tool_calls

                if text_delta:
                    existing_text = (
                        existing_text + text_delta if existing_text else text_delta
                    )

                yield JAImsStreamingMessage(
                    textDelta=text_delta,
                    message=message,
                )

    def __jaims_role_to_gemini(self, role: JAImsMessageRole) -> str:
        if role == JAImsMessageRole.USER:
            return "user"
        else:
            return "model"

    def __jaims_tools_to_gemini(
        self, jaims_tools: Optional[List[JAImsFunctionTool]] = None
    ) -> Optional[List[Tool]]:

        if not jaims_tools:
            return None

        function_declarations = []
        for jaims_tool in jaims_tools:
            function_declarations.append(
                FunctionDeclaration(
                    name=jaims_tool.descriptor.name,
                    description=jaims_tool.descriptor.description,
                    parameters=jaims_tool.descriptor.json_schema(
                        remove_any_of=True,
                        dereference=True,
                        remove_all_of=True,
                    ),
                ),
            )

        return [Tool(function_declarations=function_declarations)]

    def __jaims_messages_to_gemini(
        self, jaims_messages: List[JAImsMessage]
    ) -> tuple[List[str], ContentsType]:
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
                        parts.append(Part.from_text(content))
                    if isinstance(content, JAImsImageContent):
                        if isinstance(content.image, str):
                            raise ValueError("Image URL not supported")
                        elif isinstance(content.image, Image.Image):
                            _, data = image_to_bytes(content.image)
                            image = VertexImage.from_bytes(data)
                            parts.append(Part.from_image(image=image))
                        else:
                            raise ValueError("Invalid image content type")

            if jaims_message.tool_responses:
                for tr in jaims_message.tool_responses:
                    parts.append(
                        Part.from_function_response(
                            tr.tool_name,
                            {
                                "content": tr.response,
                            },
                        )
                    )

            if jaims_message.tool_calls:
                for tool_call in jaims_message.tool_calls:
                    parts.append(
                        Part.from_dict(
                            part_dict={
                                "function_call": {
                                    "name": tool_call.tool_name,
                                    "args": tool_call.tool_args,
                                },
                            }
                        )
                    )

            gemini_messages.append(
                Content(
                    role=role,
                    parts=parts,
                )
            )

        return system_instruction, gemini_messages

    def __gemini_to_jaims_message(
        self,
        response: GenerationResponse,
        existing_text: Optional[str] = None,
        existing_tool_calls: Optional[List[JAImsToolCall]] = None,
    ) -> Tuple[Optional[str], JAImsMessage]:
        assert response.candidates is not None

        role = JAImsMessageRole.ASSISTANT
        tool_calls = existing_tool_calls or []
        contents = []
        candidate = response.candidates[0]
        text_delta = None
        if candidate.function_calls:
            for part in candidate.content.parts:
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

        else:
            if candidate.content.text:
                text_delta = candidate.content.text
                existing_text = (
                    existing_text + text_delta if existing_text else text_delta
                )

        if existing_text:
            contents.append(existing_text)

        return text_delta, JAImsMessage(
            role=role,
            contents=contents,
            tool_calls=tool_calls,
            raw=response,
        )

    def __get_gemini_response(
        self,
        messages: Optional[List[JAImsMessage]] = None,
        tools: Optional[List[JAImsFunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
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

            if error is ValueError or not hasattr(error, "code"):
                return ErrorHandlingMethod.FAIL

            if error.code == 429 or error.code == 503 or error.code == 500:
                return ErrorHandlingMethod.EXPONENTIAL_BACKOFF

            return ErrorHandlingMethod.FAIL

        def call_gemini():
            system_instruction, gemini_messages = self.__jaims_messages_to_gemini(
                messages or []
            )
            gemini_tools = self.__jaims_tools_to_gemini(tools)

            tool_config = None
            if tool_constraints and gemini_tools:
                tool_config = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                        allowed_function_names=tool_constraints,
                    )
                )

            multimodal_model = GenerativeModel(
                self.model_name,
                generation_config=self.generation_config,
                tools=gemini_tools,
                tool_config=tool_config,
                system_instruction=system_instruction or None,  # type: ignore
            )

            return multimodal_model.generate_content(
                contents=gemini_messages,
                stream=stream,
            )

        return call_with_exponential_backoff(
            call_gemini, handle_gemini_error, self.options
        )
