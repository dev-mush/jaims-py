import os
from ...interfaces import LLMAdapterITF
from ...entities import (
    Message,
    FunctionTool,
    ToolCall,
    MessageRole,
    ImageContent,
    StreamingMessage,
    Config,
)
from ..shared.image_utilities import image_to_bytes
from ..shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
)

from typing import Iterable, List, Generator, Optional, Tuple
from PIL import Image


from google.generativeai import GenerativeModel
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
import google.ai.generativelanguage as glm


class GoogleGenerativeAIAdapter(LLMAdapterITF):
    """
    Adapter for Google Generative AI models.
    """

    def __init__(
        self,
        model: str,
        generation_config: Optional[generation_types.GenerationConfigType] = None,
        config: Optional[Config] = None,
        api_key: Optional[str] = None,
    ):
        """
        Returns a new instance of the GoogleGenerativeAIAdapter.

        Args:
            model (str): The model to use.
            generation_config (Optional[generation_types.GenerationConfigType], optional): The generation config. Defaults to None.
            config (Optional[Config], optional): The configuration options. Defaults to None.
            api_key (Optional[str], optional): The Google API key. Defaults to None.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise Exception("GOOGLE_API_KEY not provided.")

        self.model = model
        self.generation_config = generation_config
        self.config = config or Config()

    def call(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Message:
        response = self.__get_gemini_response(messages, tools, tool_constraints)
        assert isinstance(response, generation_types.GenerateContentResponse)
        _, message = self.__gemini_to_jaims_message(response)
        return message

    def call_streaming(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
        tool_constraints: Optional[List[str]] = None,
    ) -> Generator[StreamingMessage, None, None]:
        response = self.__get_gemini_response(
            messages, tools, stream=True, tool_constraints=tool_constraints
        )
        assert isinstance(response, Iterable)
        existing_text = None
        for r in response:
            if r.candidates:
                text_delta, message = self.__gemini_to_jaims_message(
                    r, existing_text=existing_text
                )
                if text_delta:
                    existing_text = (
                        existing_text + text_delta if existing_text else text_delta
                    )

                yield StreamingMessage(
                    textDelta=text_delta,
                    message=message,
                )

    def __jaims_role_to_gemini(self, role: MessageRole) -> str:
        if role == MessageRole.USER:
            return "user"
        else:
            return "model"

    def __jaims_tools_to_gemini(
        self, jaims_tools: Optional[List[FunctionTool]] = None
    ) -> Optional[List[content_types.Tool]]:

        if not jaims_tools:
            return None

        function_declarations = []
        for jaims_tool in jaims_tools:
            function_declarations.append(
                content_types.FunctionDeclaration(
                    name=jaims_tool.descriptor.name,
                    description=jaims_tool.descriptor.description,
                    parameters=jaims_tool.descriptor.json_schema(
                        remove_any_of=True,
                        dereference=True,
                        remove_all_of=True,
                    ),
                ),
            )

        return [content_types.Tool(function_declarations=function_declarations)]

    def __jaims_messages_to_gemini(
        self, jaims_messages: List[Message]
    ) -> tuple[List[str], List[content_types.ContentsType]]:
        gemini_messages = []
        system_instruction = []

        for jaims_message in jaims_messages:
            if jaims_message.role == MessageRole.SYSTEM:
                system_instruction.append(jaims_message.get_text())
                continue

            role = self.__jaims_role_to_gemini(jaims_message.role)
            parts = []

            if jaims_message.contents:
                for content in jaims_message.contents:
                    if isinstance(content, str):
                        parts.append(glm.Part(text=content))
                    if isinstance(content, ImageContent):
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

            if jaims_message.tool_responses:
                for tr in jaims_message.tool_responses:
                    parts.append(
                        glm.Part(
                            function_response=glm.FunctionResponse(
                                name=tr.tool_name,
                                response={"result": tr.response},
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
        self,
        response: generation_types.GenerateContentResponse,
        existing_text: Optional[str] = None,
    ) -> Tuple[Optional[str], Message]:
        assert response.candidates is not None

        role = MessageRole.ASSISTANT
        tool_calls = []
        contents = []
        text_delta = None
        for part in response.parts:
            if fn := part.function_call:
                args = {}
                for k, v in fn.args.items():
                    args[k] = v
                tool_calls.append(
                    ToolCall(
                        id=fn.name,
                        tool_name=fn.name,
                        tool_args=args,
                    )
                )

            if txt := part.text:
                text_delta = txt
                existing_text = existing_text + txt if existing_text else txt

        if existing_text:
            contents.append(existing_text)

        return text_delta, Message(
            role=role,
            contents=contents,
            tool_calls=tool_calls,
            raw=response,
        )

    def __get_gemini_response(
        self,
        messages: Optional[List[Message]] = None,
        tools: Optional[List[FunctionTool]] = None,
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
                tool_config = content_types.to_tool_config(
                    {
                        "function_calling_config": {
                            "mode": "any",
                            "allowed_function_names": tool_constraints,
                        }
                    }  # type: ignore
                )

            multimodal_model = GenerativeModel(
                self.model,
                generation_config=self.generation_config,
                tools=gemini_tools,
                tool_config=tool_config,
                system_instruction=system_instruction or None,
            )

            return multimodal_model.generate_content(
                contents=gemini_messages,
                stream=stream,
            )

        return call_with_exponential_backoff(
            call_gemini, handle_gemini_error, self.config
        )
