from __future__ import annotations

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Generic,
)
from typing_extensions import deprecated
from PIL import Image
from pydantic import BaseModel
import functools
import jsonref

# -------------------------
# LLM Messaging Abstraction
# -------------------------


class ToolCall:
    """
    Models a tool call requrested by an LLM.

    Some LLMs support parallel tool calls, identifying each call with a unique ID.

    Attributes:
        id (str): The ID of the tool call.
        tool_name (str): The name of the tool being called.
        tool_args (Optional[Dict[str, Any]]): The arguments for the tool call.
    """

    def __init__(self, id: str, tool_name: str, tool_args: Optional[Dict[str, Any]]):
        self.id = id
        self.tool_name = tool_name
        self.tool_args = tool_args


@deprecated("use ToolCall instead.")
class JAImsToolCall(ToolCall):
    pass


class ToolResponse:
    """
    Models a response from a ToolCall.

    In case of parallel tool calls, the response is associated with the tool call ID.

    Attributes:
        tool_call_id (str): The ID of the tool call.
        tool_name (str): The name of the tool.
        response (Any): The response from the tool call.
        is_error (bool): Whether the response is an error.
    """

    def __init__(
        self, tool_call_id: str, tool_name: str, response: Any, is_error: bool = False
    ):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.response = response
        self.is_error = is_error


@deprecated("use ToolResponse instead.")
class JAImsToolResponse(ToolResponse):
    pass


ImageContentType = Union[str, Image.Image]


class ImageContent:
    """
    Models an image content for a message that contains an image.

    It can be either a PIL Image object or a URL to the image.

    Attributes:
        image (ImageContentType): The image content.
    """

    def __init__(self, image: ImageContentType):
        self.image = image


ContentType = Union[ImageContent, str]


class MessageRole(Enum):
    """
    Represents the roles of a Message sent and received by an LLM.

    Roles have different meanings depending on the LLM used, this is an abstraction over the most common approaches.
    It is mapped back by the adapter to the appropriate role based on the LLM implementation.

    The MessageRole enum defines the different roles that a message can have in the JAIms domain. Meaning and usage may vary depending on the adapter being used.

    The available roles are:
    - USER: Represents a message sent by a user.
    - ASSISTANT: Represents a message sent by the assistant.
    - TOOL: Represents a message sent by a tool.
    - SYSTEM: Represents a system-instruction message.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Message:
    """
    Represents a message to be sent to or received from an LLM.

    Messages can have different types of content depending on the multimodality of the LLM.
    A message can have also associated tool calls if an LLM has requested tool invocation, or a tool_response when a tool has been invoked. The tool_calls are represented as an array because they can be invoked in parallel when the LLM supports it.

    Raw contains, depending on the adapter used, the raw response from the LLM used.

    Attributes:
        role (MessageRole): The role of the message (USER, ASSISTANT, SYSTEM, TOOL).
        contents (Optional[List[ContentType]]): The contents of the message.
        name (Optional[str]): The name associated with the message.
        tool_calls (Optional[List[ToolCall]]): The tool calls associated with the message.
        tool_responses (Optional[List[ToolResponse]]): The tool responses associated with a message.
        raw (Optional[Any]): The raw data associated with the message.
    """

    def __init__(
        self,
        role: MessageRole,
        contents: Optional[List[ContentType]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_responses: Optional[List[ToolResponse]] = None,
        raw: Optional[Any] = None,
    ):
        self.role = role
        self.contents = contents
        self.name = name
        self.tool_calls = tool_calls
        self.tool_responses = tool_responses
        self.raw = raw

    def get_text(self) -> Optional[str]:
        """
        Get the text content of the message.

        When the message has multiple textual contents, they are concatenated.

        Returns:
            Optional[str]: The text content of the message, or None if there is no content.
        """
        if self.contents is None:
            return None

        all_text = ""
        for content in self.contents:
            if isinstance(content, str):
                all_text += content

        if all_text:
            return all_text

        return None

    # -------------------------
    # convenience constructors
    # -------------------------

    @staticmethod
    def user_message(
        text: str,
        raw: Optional[Any] = None,
    ) -> Message:
        """
        Create a user message.

        Args:
            text (str): The text content of the message.
            raw (Optional[Any]): The raw data associated with the message.

        Returns:
            Message: The created user message.
        """
        return Message(
            role=MessageRole.USER,
            contents=[text],
            raw=raw,
        )

    @staticmethod
    def assistant_message(text: str, raw: Optional[Any] = None) -> Message:
        """
        Create an assistant message.

        Args:
            text (str): The text content of the message.
            raw (Optional[Any]): The raw data associated with the message.

        Returns:
            Message: The created assistant message.
        """
        return Message(
            role=MessageRole.ASSISTANT,
            contents=[text],
            raw=raw,
        )

    @staticmethod
    def system_message(text: str, raw: Optional[Any] = None) -> Message:
        """
        Create a system message.

        Args:
            text (str): The text content of the message.
            raw (Optional[Any]): The raw data associated with the message.

        Returns:
            Message: The created system message.
        """
        return Message(
            role=MessageRole.SYSTEM,
            contents=[text],
            raw=raw,
        )

    @staticmethod
    def tool_response_message(
        responses: List[ToolResponse], raw: Optional[Any] = None
    ) -> Message:
        """
        Create a tool response message.

        Args:
            responses (List[ToolResponse]): The tool responses associated with the message.
            raw (Optional[Any]): The raw data associated with the message.

        Returns:
            Message: The created tool response message.
        """
        return Message(
            role=MessageRole.TOOL,
            tool_responses=responses,
            raw=raw,
        )

    @staticmethod
    def tool_call_message(
        tool_calls: List[ToolCall], raw: Optional[Any] = None
    ) -> Message:
        """
        Create a tool call message.

        Args:
            tool_calls (List[ToolCall]): The tool calls associated with the message.
            raw (Optional[Any]): The raw data associated with the message.

        Returns:
            Message: The created tool call message.
        """
        return Message(
            role=MessageRole.ASSISTANT,
            tool_calls=tool_calls,
            raw=raw,
        )


@deprecated("use Message instead.")
class JAImsMessage(Message):
    pass


class StreamingMessage:
    """
    Represents the streaming message being received by an LLM.

    The message attribute contains the complete message being receved by the LLM, while the textDelta that is being streamed.
    When reading the stream, you're usually interested in printing the textDelta, and once the stream is complete, you can use the message attribute to get the full message object.

    Attributes:
        message (Message): The main message object.
        textDelta (Optional[str]): A delta fragment of the message being streamed.
    """

    def __init__(self, message: Message, textDelta: Optional[str] = None):
        self.message = message
        self.textDelta = textDelta


# -----------------------------------
# Tools and function handling classes
# -----------------------------------

ModelT = TypeVar("ModelT", bound=BaseModel)


class FunctionToolDescriptor(Generic[ModelT]):
    """
    Models a function tool that an LLM interacts with.

    This class is used to ensure that a function tool is parsed correctly for the LLM. A function tool is understood by an LLM generally trough its name, description, and a schema for the expected parameters.

    The choice here has been to adopt pydantic BaseModel to ensure that the schema is correctly parsed and validated.
    A function tool can also not have any parameters, in which case the params attribute is None.

    Attributes:
        name (str): The name of the function tool.
        description (str): The description of the function tool.
        params (Optional[type[BaseModel]]): The expected parameters for the function tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        params: Optional[type[ModelT]],
    ):
        self.name = name
        self.description = description
        self.params = params

    def json_schema(
        self,
        remove_titles: bool = True,
        remove_any_of: bool = False,
        remove_all_of: bool = False,
        dereference: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns the JSON schema for the entity.

        If there are no parameters, an empty dictionary is returned.

        Args:
            remove_titles (bool): Whether to remove the title key from the schema. Defaults to True.
            remove_any_of (bool): Whether to remove the anyOf key from the schema when multiple types are present. Defaults to False.
            remove_all_of (bool): Whether to remove the allOf key from the schema when multiple types are present. Defaults to False.
            dereference (bool): Whether to dereference the schema. Defaults to False.

        Returns:
            A dictionary representing the JSON schema for the entity.
        """
        if not self.params:
            return {}

        schema = self.params.model_json_schema()

        if dereference:
            schema = self.__dereference_schema(schema)

        if remove_any_of:
            schema = self.__remove_any_of_for_multiple_types(schema)

        if remove_all_of:
            schema = self.__remove_all_of_for_multiple_types(schema)

        if remove_titles:
            schema = self.__remove_title_from_schema(schema)

        return schema

    def __remove_title_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively removes the "title" key from the schema.

        Args:
            schema (Dict[str, Any]): The schema to remove the title from.

        Returns:
            Dict[str, Any]: The schema without the title key.
        """
        if not isinstance(schema, dict):
            return schema

        return {
            k: self.__remove_title_from_schema(v)
            for k, v in schema.items()
            if k != "title"
        }

    def __remove_any_of_for_multiple_types(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively removes the "anyOf" key from the schema when multiple types are present.

        Args:
            schema (Dict[str, Any]): The schema to remove the anyOf key from.

        Returns:
            Dict[str, Any]: The schema without the anyOf key.
        """

        def walk_properties(properties):
            for key, value in properties.items():
                if isinstance(value, dict):
                    if "anyOf" in value:
                        first_values_domain = value["anyOf"][0]
                        for k, v in first_values_domain.items():
                            value[k] = v
                        del value["anyOf"]
                    walk_properties(value)

        walk_properties(schema)

        return schema

    def __remove_all_of_for_multiple_types(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively removes "allOf" keys from the schema when multiple types are present.

        Args:
            schema (Dict[str, Any]): The schema to remove the allOf key from.

        Returns:
            Dict[str, Any]: The schema without the allOf key.
        """

        def walk_properties(properties):
            for _, value in properties.items():
                if isinstance(value, dict):
                    if "allOf" in value:
                        first_values_domain = value["allOf"][0]
                        for k, v in first_values_domain.items():
                            value[k] = v
                        del value["allOf"]
                    walk_properties(value)

        walk_properties(schema)

        return schema

    def __dereference_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively dereferences the schema.

        Args:
            schema (Dict[str, Any]): The schema to dereference.

        Returns:
            Dict[str, Any]: The dereferenced schema.
        """
        dereferenced_schema: dict = jsonref.replace_refs(schema)  # type: ignore

        return {
            "type": dereferenced_schema["type"],
            "properties": dereferenced_schema["properties"],
            "required": dereferenced_schema.get("required", []),
        }


@deprecated("use FunctionToolDescriptor instead.")
class JAImsFunctionToolDescriptor(FunctionToolDescriptor):
    pass


class FunctionTool:
    """
    Wrapper class that binds a python function to a FunctionToolDescriptor

    This class supports both simple function and class methods, and implements the __call__ method to seamlessly call the wrapped function.

    When directly invoked as function, the class simply forwards the call to the wrapped function, when called trough the call_raw method, it parses the raw data into the expected parameters defined in the descriptor and forwards the formatted data to the wrapped function.

    By default, the class uses a simple default formatter that assumes that the wrapped function expects as argument the parsed data from the descriptor, and no keyword arguments, use a custom formatter to handle more complex cases.

    Also take a look at the jaimsfunctiontool decorator that lets you easily decorate a function to be used seamlessly as a function tool (supporting basic python types and pydantic models as parameters).

    Args:
        descriptor (FunctionToolDescriptor): The descriptor for the function tool.
        function (Optional[Callable]): The function to be wrapped by the tool. Defaults to None.
        formatter (Optional[Callable[[Dict], Tuple[tuple, dict]]]): The formatter function to parse the data. Defaults to None (uses the default formatter).
    """

    def __init__(
        self,
        descriptor: FunctionToolDescriptor[ModelT],
        function: Optional[Callable] = None,
        formatter: Optional[Callable[[Dict], Tuple[tuple, dict]]] = None,
    ):
        self.descriptor = descriptor
        self.formatter = formatter or self.__default_formatter
        self.__bound_instance = None
        self.function = function

        if function:
            functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        """
        Calls the wrapped function with the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the function call.
        """
        if not self.function:
            return None

        # handle case when wrapped function is an instance method
        if self.__bound_instance:
            return self.function(self.__bound_instance, *args, **kwargs)

        return self.function(*args, **kwargs)

    def __get__(self, instance, owner):
        """
        This is a descriptor method that allows the class to be used as a decorator.
        In this case, it captures the instance to which the class is bound so that it can pass it to the wrapped function in case the function is an instance method.
        """
        self.__bound_instance = instance
        return self

    def __default_formatter(self, data: Dict) -> Tuple[tuple, dict]:
        """
        Default formatter function to parse the data.

        Args:
            data (Dict): The data to be formatted.

        Returns:
            Tuple[tuple, dict]: The formatted arguments and keyword arguments.

        Raises:
            ValueError: If the data does not match the expected parameters defined in the descriptor.
        """
        if not self.descriptor.params or self.function is None:
            return ((), data)

        try:
            parsed_args = self.descriptor.params.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid parameters for tool {self.descriptor.name}: {e}")

        return (parsed_args,), {}

    def call_raw(self, **kwargs):
        """
        Calls the wrapped function parsing the raw data (received from the LLM) into the expected parameters defined in the descriptor.

        Args:
            kwargs (dict): The raw data received from the LLM.

        Returns:
            Any: The result of the function call.

        Raises:
            ValueError: If the data does not match the expected parameters defined in the descriptor.
        """
        if not self.function:
            return None

        args, kwargs = self.formatter(kwargs)

        return self(*args, **kwargs)


@deprecated("use FunctionTool instead.")
class JAImsFunctionTool(FunctionTool):
    pass


# -----------------------------------
# LLM Model Configuration and Options
# -----------------------------------


class LLMParams:
    """
    Models common configuration parameters sent to the LLM.

    This class represents an abstraction over the most common configuration parameters for an LLM.
    Some LLMs support response formatting specifications, use the appropriate dictionary to specify the format depending on the LLM implementation being used.

    Args:
        temperature (float, optional): The temperature parameter for generating responses. Defaults to 0.5.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.
        response_format (Optional[Dict[str, Any]], optional): The format of the response. Defaults to None.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format


@deprecated("use LLMParams instead.")
class JAImsLLMConfig(LLMParams):
    pass


class Config:
    """
    Config options for the Agent class when calling the remote LLM.

    These are mainly client options that control the behavior of the client when calling the LLM API and in case of errors or timeouts.
    JAIms natively supports exponential backoff for retries, and the options here allow to configure the behavior of the backoff.
    Exponential backoff is calculated using the formula: min(delay * exponential_base, exponential_cap) * (1 + jitter * random())

    Args:
        max_retries (int): The maximum number of retries after a failing a call.
        retry_delay (int): The delay between each retry in case of failure without exponential backoff.
        exponential_base (int): The base for exponential backoff calculation.
        exponential_delay (int): The initial delay, in seconds, to multiply by the base for exponential backoff.
        exponential_cap (Optional[int]): The maximum value, in seconds, for exponential backoff delay. Leave None to let it grow indefinitely.
        jitter (bool): Whether to add a small jitter to the delay (to avoid concurrent firing), in case of exponential backoff, in the worst case, it will be 2x the delay.
        platform_specific_options (Optional[Dict[str, Any]]): Platform-specific options to be passed to the client.
    """

    def __init__(
        self,
        max_retries=15,
        retry_delay=10,
        exponential_base: int = 2,
        exponential_delay: int = 1,
        exponential_cap: Optional[int] = None,
        jitter: bool = True,
        platform_specific_options: Optional[Dict[str, Any]] = None,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_base = exponential_base
        self.exponential_delay = exponential_delay
        self.exponential_cap = exponential_cap
        self.jitter = jitter
        self.platform_specific_options = platform_specific_options or {}

    def copy_with_overrides(
        self,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        exponential_base: Optional[int] = None,
        exponential_delay: Optional[int] = None,
        exponential_cap: Optional[int] = None,
        jitter: Optional[bool] = None,
        platform_specific_options: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """
        Returns a new Config instance with the passed parameters overridden.
        """
        return Config(
            max_retries=max_retries if max_retries else self.max_retries,
            retry_delay=retry_delay if retry_delay else self.retry_delay,
            exponential_base=(
                exponential_base if exponential_base else self.exponential_base
            ),
            exponential_delay=(
                exponential_delay if exponential_delay else self.exponential_delay
            ),
            exponential_cap=(
                exponential_cap if exponential_cap else self.exponential_cap
            ),
            jitter=jitter if jitter else self.jitter,
            platform_specific_options=(
                platform_specific_options
                if platform_specific_options
                else self.platform_specific_options
            ),
        )


@deprecated("use Config instead.")
class JAImsOptions(Config):
    pass


# ----------
# exceptions
# ----------


class MaxConsecutiveFunctionCallsExceeded(Exception):
    """
    Exception raised when the maximum number of consecutive function calls is exceeded.

    Attributes:
        max_consecutive_calls -- maximum number of consecutive calls allowed
    """

    def __init__(self, max_consecutive_calls):
        message = f"Max consecutive function calls exceeded: {max_consecutive_calls}"
        super().__init__(message)


class UnexpectedFunctionCall(Exception):
    """
    Exception raised when an unexpected function call occurs.

    Attributes:
        func_name -- name of the unexpected function
    """

    def __init__(self, func_name):
        message = f"Unexpected function call: {func_name}"
        super().__init__(message)


class MaxRetriesExceeded(Exception):
    """
    Exception raised when the maximum number of retries is performed by an adapter client.

    Attributes:
        max_consecutive_calls -- maximum number of consecutive calls allowed
    """

    def __init__(self, max_retries, last_exception=None):
        message = (
            f"Max retries exceeded: {max_retries}, last exception: {last_exception}"
        )
        super().__init__(message)
