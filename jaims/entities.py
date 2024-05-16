from __future__ import annotations

# Enum class over all Json Types
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


# -------------------------
# LLM Messaging Abstraction
# -------------------------


class JAImsToolCall:
    def __init__(self, id: str, tool_name: str, tool_args: Optional[Dict[str, Any]]):
        self.id = id
        self.tool_name = tool_name
        self.tool_args = tool_args


class JAImsToolResponse:
    def __init__(self, tool_call_id: str, tool_name: str, response: Any):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.response = response


class JAImsContentTypes(Enum):
    TEXT = "text"
    IMAGE = "image"


class JAImsMessageContent:
    def __init__(self, content: Any, type: JAImsContentTypes) -> None:
        self.type = type
        self.content = content


class JAImsMessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class JAImsMessage:
    def __init__(
        self,
        role: JAImsMessageRole,
        contents: Optional[List[JAImsMessageContent]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[JAImsToolCall]] = None,
        tool_response: Optional[JAImsToolResponse] = None,
        raw: Optional[Any] = None,
    ):
        self.role = role
        self.contents = contents
        self.name = name
        self.tool_calls = tool_calls
        self.tool_response = tool_response
        self.raw = raw

    def get_text(self) -> str:
        if self.contents is None:
            return ""
        return "".join(
            c.content for c in self.contents if c.type == JAImsContentTypes.TEXT
        )

    # -------------------------
    # convenience constructors
    # -------------------------

    @staticmethod
    def user_message(text: str) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.USER,
            contents=[JAImsMessageContent(content=text, type=JAImsContentTypes.TEXT)],
        )

    @staticmethod
    def assistant_message(text: str, raw: Optional[Any]) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.ASSISTANT,
            contents=[JAImsMessageContent(content=text, type=JAImsContentTypes.TEXT)],
            raw=raw,
        )

    @staticmethod
    def system_message(text: str, raw: Optional[Any]) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.SYSTEM,
            contents=[JAImsMessageContent(content=text, type=JAImsContentTypes.TEXT)],
            raw=raw,
        )

    @staticmethod
    def tool_response_message(
        tool_call_id: str, tool_name: str, response: Any, raw: Optional[Any] = None
    ) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.TOOL,
            tool_response=JAImsToolResponse(tool_call_id, tool_name, response),
            raw=raw,
        )

    @staticmethod
    def tool_call_message(
        tool_calls: List[JAImsToolCall], raw: Optional[Any] = None
    ) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.ASSISTANT,
            tool_calls=tool_calls,
            raw=raw,
        )


class JAImsStreamingMessage:
    def __init__(self, message: JAImsMessage, textDelta: Optional[str] = None):
        self.message = message
        self.textDelta = textDelta


# ------------------------------------------
# Params, tool and function handling classes
# ------------------------------------------


class JAImsJsonSchemaType(Enum):
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    ARRAY = "array"
    BOOLEAN = "boolean"
    NULL = "null"


class JAImsParamDescriptor:
    """
    Describes a parameter to be used in the OPENAI API.

    Attributes
    ----------
        name : str
            the parameter name
        description : str
            the parameter description
        json_type : JsonType:
            the parameter json type
        attributes_params_descriptors : list of JAImsParamDescriptor
            the list of parameters descriptors for the attributes of the parameter
            in case the parameter is an object, defualts to None
        array_type_descriptors : list of JAImsParamDescriptor
            the parameter descriptors for the array type in case the parameter is an array, defaults to None
        enum_values:
            the list of values in case the parameter is an enum, defaults to None
        required : bool
            whether the parameter is required or not, defaults to True

    """

    def __init__(
        self,
        name: str,
        description: str,
        json_type: JAImsJsonSchemaType,
        attributes_params_descriptors: Optional[List[JAImsParamDescriptor]] = None,
        array_type_descriptors: Optional[List[JAImsParamDescriptor]] = None,
        array_type_any_valid: bool = True,
        enum_values: Optional[List[Any]] = None,
        required: bool = True,
    ):
        self.name = name
        self.description = description
        self.json_type = json_type
        self.attributes_params_descriptors = attributes_params_descriptors
        self.array_type_descriptors = array_type_descriptors
        self.array_type_any_valid = array_type_any_valid
        self.enum_values = enum_values
        self.required = required

    def get_jsonapi_schema(self) -> Dict[str, Any]:
        """
        Returns the jsonapi schema for the parameter.
        """
        schema: dict[str, Any] = {
            "type": self.json_type.value,
            "description": self.description,
        }

        if (
            self.json_type == JAImsJsonSchemaType.OBJECT
            and self.attributes_params_descriptors
        ):
            schema["properties"] = {}
            schema["required"] = []
            for param in self.attributes_params_descriptors:
                schema["properties"][param.name] = param.get_jsonapi_schema()
                if param.required:
                    schema["required"].append(param.name)

        if self.json_type == JAImsJsonSchemaType.ARRAY and self.array_type_descriptors:
            items_schema = [
                desc.get_jsonapi_schema() for desc in self.array_type_descriptors
            ]
            if self.array_type_any_valid:
                schema["items"] = {"anyOf": items_schema}
            else:
                schema["items"] = [items_schema]

        if self.enum_values:
            schema["enum"] = self.enum_values

        return schema


class JAImsFunctionToolDescriptor:
    """
    Describes a tool to be used in the OPENAI API. Supports only function tool for now.

    Attributes
    ----------
        name : str
            the tool name
        description : str
            the tool description
        params_descriptors: List[JAImsParamDescriptor]
            the list of parameters descriptors
    """

    def __init__(
        self,
        name: str,
        description: str,
        params_descriptors: List[JAImsParamDescriptor],
    ):
        self.name = name
        self.description = description
        self.params_descriptors = params_descriptors

    def get_jsonapi_schema(self) -> Dict[str, Any]:
        """
        Returns the jsonapi schema for function.
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in self.params_descriptors:
            schema["properties"][param.name] = param.get_jsonapi_schema()
            if param.required:
                schema["required"].append(param.name)

        return schema


class JAImsFunctionTool:
    """
    Wraps a function tool used by the LLM along with a function to be called locally when
    the tool is invoked by the LLM.
    You may subclass this class to implement your own function wrapper behavior by overriding the call method.


    Attributes
    ----------
        function : Callable[..., Any]
            The function to be called when the tool is invoked, defaults to None.
            When None, the tool call pass None to the agent as a result.
        to_openai_function_tool : JAImsFunctionToolDescriptor
            The tool descriptor, contains the markup information that will be used to be passed
            as a tool invocation dictionary to the LLM.

    Methods
    -------
        call(params: Dict[str, Any]) -> Any
            Calls the wrapped function with the given parameters, if the function is not None.
            Returns None otherwise.
    """

    def __init__(
        self,
        function_tool_descriptor: JAImsFunctionToolDescriptor,
        function: Optional[Callable[..., Any]] = None,
    ):
        self.function = function
        self.function_tool = function_tool_descriptor

    def call(self, params: Optional[Dict[str, Any]]) -> Any:
        """
        Calls the wrapped function with the passed parameters if the function is not None.
        Returns None otherwise.

        Parameters
        ----------
            params : dict
                the parameters passed to the wrapped function
        """

        return self.function(**params) if params and self.function else None


# ----------
# exceptions
# ----------


class JAImsTokensLimitExceeded(Exception):
    """
    Exception raised when the token limit is exceeded.

    Attributes:
        max_tokens -- maximum number of tokens allowed
        messages_tokens -- number of tokens in the messages
        llm_buffer -- buffer for the LLM answer
        has_optimized -- flag indicating if the messages have been optimized
    """

    def __init__(self, max_tokens, messages_tokens, llm_buffer, has_optimized):
        message = f"Max tokens: {max_tokens}\n LLM Answer Buffer: {llm_buffer}\n Messages tokens: {messages_tokens}\n Messages Optimized: {has_optimized}  "
        super().__init__(message)


class JAImsMissingOpenaiAPIKeyException(Exception):
    """
    Exception raised when the OPENAI_API_KEY is missing.
    """

    def __init__(self):
        message = "Missing OPENAI_API_KEY, set environment variable OPENAI_API_KEY or pass it as a parameter to the agent constructor."
        super().__init__(message)


class JAImsOpenAIErrorException(Exception):
    """
    Exception raised when there is an error with OpenAI.

    Attributes:
        message -- explanation of the error
        openai_error -- the error from OpenAI
    """

    def __init__(self, message, openai_error):
        super().__init__(message)
        self.openai_error = openai_error


class JAImsMaxConsecutiveFunctionCallsExceeded(Exception):
    """
    Exception raised when the maximum number of consecutive function calls is exceeded.

    Attributes:
        max_consecutive_calls -- maximum number of consecutive calls allowed
    """

    def __init__(self, max_consecutive_calls):
        message = f"Max consecutive function calls exceeded: {max_consecutive_calls}"
        super().__init__(message)


class JAImsUnexpectedFunctionCall(Exception):
    """
    Exception raised when an unexpected function call occurs.

    Attributes:
        func_name -- name of the unexpected function
    """

    def __init__(self, func_name):
        message = f"Unexpected function call: {func_name}"
        super().__init__(message)
