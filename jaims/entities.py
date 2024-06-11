from __future__ import annotations

# Enum class over all Json Types
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field


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


JAImsImageContentType = Union[str, Image.Image]


class JAImsImageContent:
    def __init__(self, image: JAImsImageContentType):
        self.image = image


JAImsContentType = Union[JAImsImageContent, str]


class JAImsMessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class JAImsMessage:
    def __init__(
        self,
        role: JAImsMessageRole,
        contents: Optional[List[JAImsContentType]] = None,
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

    def get_text(self) -> Optional[str]:
        if self.contents is None:
            return None

        all_text = ""
        for content in self.contents:
            if isinstance(content, str):
                all_text += content

        return all_text or None

    # -------------------------
    # convenience constructors
    # -------------------------

    @staticmethod
    def user_message(
        text: str,
        raw: Optional[Any] = None,
    ) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.USER,
            contents=[text],
            raw=raw,
        )

    @staticmethod
    def assistant_message(text: str, raw: Optional[Any] = None) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.ASSISTANT,
            contents=[text],
            raw=raw,
        )

    @staticmethod
    def system_message(text: str, raw: Optional[Any] = None) -> JAImsMessage:
        return JAImsMessage(
            role=JAImsMessageRole.SYSTEM,
            contents=[text],
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


# -----------------------------------
# Tools and function handling classes
# -----------------------------------


class JAImsFunctionToolDescriptor:
    """
    Describes a function tool.

    Attributes
    ----------
        name : str
            the tool name
        description : str
            the tool description
        params: a BaseModel type
            the tool parameters
    """

    def __init__(
        self,
        name: str,
        description: str,
        params: Optional[type[BaseModel]],
    ):
        self.name = name
        self.description = description
        self.params = params

    def json_schema(self) -> Dict[str, Any]:

        if not self.params:
            return {}

        return self.params.model_json_schema()


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
        descriptor : JAImsFunctionToolDescriptor
            The tool descriptor, contains the markup information that will be used to be passed
            as a tool invocation dictionary to the LLM.
    """

    def __init__(
        self,
        descriptor: JAImsFunctionToolDescriptor,
        function: Optional[Callable[..., Any]] = None,
    ):
        self.function = function
        self.descriptor = descriptor

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.descriptor.params:
            return

        try:
            parsed_args = self.descriptor.params.model_validate(*args)
        except Exception as e:
            raise ValueError(f"Invalid parameters for tool {self.descriptor.name}: {e}")

        return self.function(parsed_args) if self.function else None

    def call(self, *args: Any, **kwds: Any) -> Any:
        """
        Calls the function with the provided arguments.
        """
        return self(*args, **kwds)


# -----------------------------------
# LLM Model Configuration and Options
# -----------------------------------


class JAImsModelCode:
    """
    Constants for the LLM names.
    """

    GEMINI_1_PRO = "gemini-1.0-pro"
    GEMINI_1_PRO_LATEST = "gemini-1.0-pro-latest"
    GEMINI_1_PRO_001 = "gemini-1.0-pro-001"
    GEMINI_1_PRO_VISION = "gemini-1.0-pro-vision"
    GEMINI_1_PRO_VISION_LATEST = "gemini-1.0-pro-vision-latest"
    GEMINI_1_5_FLASH_LATEST = "gemini-1.5-flash-latest"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_PRO_LATEST = "gemini-1.5-pro-latest"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K_0613 = "gpt-4-32k-0613"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT_4_1106_VISION_PREVIEW = "gpt-4-1106-vision-preview"
    GPT_4o = "gpt-4o"
    GPT_4o_2024_05_13 = "gpt-4o-2024-05-13"


class JAImsLLMConfig:
    def __init__(
        self,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format


class JAImsOptions:
    """
    Config options for the JAImsAgent when calling the remote LLM.
    Exponential backoff is calculated using the formula: min(delay * exponential_base, exponential_cap) * (1 + jitter * random())

    Args:
        max_retries (int): The maximum number of retries after a failing a call.
        retry_delay (int): The delay between each retry in case of failure without exponential backoff.
        exponential_base (int): The base for exponential backoff calculation.
        exponential_delay (int): The initial delay, in seconds, to multiply by the base for exponential backoff.
        exponential_cap (Optional[int]): The maximum value, in seconds, for exponential backoff delay. Leave None to let it grow indefinitely.
        jitter (bool): Whether to add a small jitter to the delay (to avoid concurrent firing), in case of exponential backoff, in the worst case, it will be 2x the delay.
    """

    def __init__(
        self,
        max_retries=15,
        retry_delay=10,
        exponential_base: int = 2,
        exponential_delay: int = 1,
        exponential_cap: Optional[int] = None,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_base = exponential_base
        self.exponential_delay = exponential_delay
        self.exponential_cap = exponential_cap
        self.jitter = jitter

    def copy_with_overrides(
        self,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        exponential_base: Optional[int] = None,
        exponential_delay: Optional[int] = None,
        exponential_cap: Optional[int] = None,
        jitter: Optional[bool] = None,
    ) -> JAImsOptions:
        """
        Returns a new JAImsOptions instance with the passed parameters overridden.
        """
        return JAImsOptions(
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
        )


# ----------
# exceptions
# ----------


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


class JAImsMaxRetriesExceeded(Exception):
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
