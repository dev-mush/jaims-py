from __future__ import annotations

# Enum class over all Json Types
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


# ---------
# constants
# ---------


DEFAULT_MAX_TOKENS = 1024
MAX_CONSECUTIVE_CALLS = 10

# ---------------------
# openai / LLM modeling
# ---------------------


class JAImsGPTModel(Enum):
    """
    The OPENAI GPT models available.
    Only those that support functions are listed, so just:
    gpt-3.5-turbo-0613, gpt-3-5-turbo-16k-0613, gpt-4-0613
    """

    GPT_3_5_TURBO = ("gpt-3.5-turbo", 4096, 0.0015, 0.002)
    GPT_3_5_TURBO_16K = ("gpt-3.5-turbo-16k", 16384, 0.003, 0.004)
    GPT_3_5_TURBO_0613 = ("gpt-3.5-turbo-0613", 4096, 0.0015, 0.002)
    GPT_3_5_TURBO_16K_0613 = ("gpt-3.5-turbo-16k-0613", 16384, 0.003, 0.004)
    GPT_3_5_TURBO_1106 = ("gpt-3.5-turbo-1106", 16385, 0.001, 0.002)
    GPT_4 = ("gpt-4", 8192, 0.03, 0.06)
    GPT_4_32K = ("gpt-4-32k", 32768, 0.06, 0.12)
    GPT_4_0613 = ("gpt-4-0613", 8192, 0.03, 0.06)
    GPT_4_32K_0613 = ("gpt-4-32k-0613", 32768, 0.06, 0.12)
    GPT_4_1106_PREVIEW = ("gpt-4-1106-preview", 128000, 0.01, 0.03)
    GPT_4_VISION_PREVIEW = ("gpt-4-vision-preview", 128000, 0.01, 0.03)

    def __init__(self, string, max_tokens, price_1k_tokens_in, price_1k_tokens_out):
        self.string = string
        self.max_tokens = max_tokens
        self.price_1k_tokens_in = price_1k_tokens_in
        self.price_1k_tokens_out = price_1k_tokens_out

    def __str__(self):
        return self.string


class JAImsTokensExpense:
    """
    Tracks the number of tokens spent on a job and on which GPTModel.
    """

    def __init__(
        self,
        gpt_model: JAImsGPTModel,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        rough_estimate=False,
    ):
        self.gpt_model = gpt_model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.rough_estimate = rough_estimate

    @staticmethod
    def from_openai_usage_dictionary(
        gpt_model: JAImsGPTModel, dictionary: dict
    ) -> JAImsTokensExpense:
        return JAImsTokensExpense(
            gpt_model=gpt_model,
            prompt_tokens=dictionary["prompt_tokens"],
            completion_tokens=dictionary["completion_tokens"],
            total_tokens=dictionary["total_tokens"],
        )

    def spend(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def add_from(self, other_expense: JAImsTokensExpense):
        self.prompt_tokens += other_expense.prompt_tokens
        self.completion_tokens += other_expense.completion_tokens
        self.total_tokens += other_expense.total_tokens
        if other_expense.rough_estimate:
            self.rough_estimate = True  # becomes rough if summed with something rough

    def get_cost(self):
        return (self.prompt_tokens / 1000) * self.gpt_model.price_1k_tokens_in + (
            self.completion_tokens / 1000
        ) * self.gpt_model.price_1k_tokens_out

    def __str__(self):
        string_repr = (
            f"GPT model: {self.gpt_model}\n"
            f"Prompt tokens: {self.prompt_tokens}\n"
            f"Completion tokens: {self.completion_tokens}\n"
            f"Total tokens: {self.total_tokens}\n"
            f"Cost: {round(self.get_cost(),4)}$"
        )

        if self.rough_estimate:
            string_repr += "\n(warning: rough estimate)"

        return string_repr

    def to_json(self):
        return {
            "model": self.gpt_model.string,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.get_cost(),
            "rough_estimate": self.rough_estimate,
        }


class JAImsOpenaiKWArgs:
    """
    Represents the keyword arguments for the JAIms OpenAI wrapper.
    This class entirely mirrors the openai API parameters, so refer to it for documentation.
    (https://platform.openai.com/docs/api-reference/chat/create).

    Args:
        model (JAImsGPTModel, optional): The OpenAI model to use. Defaults to JAImsGPTModel.GPT_3_5_TURBO.
        messages (List[dict], optional): The list of messages for the chat completion. Defaults to an empty list, it is automatically populated by the run method so it is not necessary to pass them. If passed, they will always be appended to the messages passed in the run method.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 500.
        stream (bool, optional): Whether to use streaming for the API call. Defaults to False.
        temperature (float, optional): The temperature for generating creative text. Defaults to 0.0.
        top_p (Optional[int], optional): The top-p value for nucleus sampling. Defaults to None.
        n (int, optional): The number of responses to generate. Defaults to 1.
        seed (Optional[int], optional): The seed to be passed to openai to have more consistent outputs. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for avoiding repetitive responses. Defaults to 0.0.
        presence_penalty (float, optional): The presence penalty for encouraging diverse responses. Defaults to 0.0.
        logit_bias (Optional[Dict[str, float]], optional): The logit bias for influencing the model's output. Defaults to None.
        response_format (Optional[Dict], optional): The format for the generated response. Defaults to None.
        stop (Union[Optional[str], Optional[List[str]]], optional): The stop condition for the generated response. Defaults to None.
        tool_choice (Union[str, Dict], optional): The choice of tool to use. Defaults to "auto".
        tools (Optional[List[JAImsFunctionToolWrapper]], optional): The list of function tool wrappers to use. Defaults to None.
    """

    def __init__(
        self,
        model: JAImsGPTModel = JAImsGPTModel.GPT_3_5_TURBO,
        messages: List[dict] = [],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        temperature: float = 0.0,
        top_p: Optional[int] = None,
        n: int = 1,
        seed: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        response_format: Optional[Dict] = None,
        stop: Union[Optional[str], Optional[List[str]]] = None,
        tool_choice: Union[str, Dict] = "auto",
        tools: Optional[List[JAImsFuncWrapper]] = None,
    ):
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias
        self.response_format = response_format
        self.stop = stop
        self.tool_choice = tool_choice
        self.tools = tools

    def to_dict(self):
        kwargs = {
            "model": self.model.string,
            "temperature": self.temperature,
            "n": self.n,
            "stream": self.stream,
            "messages": self.messages,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "stop": self.stop,
        }

        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        if self.logit_bias:
            kwargs["logit_bias"] = self.logit_bias

        if self.tools:
            kwargs["tools"] = [
                tool.function_tool.to_openai_function_tool() for tool in self.tools
            ]
            kwargs["tool_choice"] = self.tool_choice

        return kwargs


class JAImsOptions:
    """
    Represents the options for JAImsAgent.

    Args:
        leading_prompts (Optional[List[Dict]]): A list of leading prompts, these will be always prepended to the history for each run.
        trailing_prompts (Optional[List[Dict]]): A list of trailing promptsm, these will be always appended to the history for each run.
        max_consecutive_function_calls (int): The maximum number of consecutive function calls allowed (defaults to 10 to avoid infinite loops).
        optimize_context (bool): Whether to optimize the context in the history manager or not, defaults to True.
        message_history_size (Optional[int]): The size of the message history for each run, only the last n messages will be passed, defaults to none (every message is passed until optimization starts).
        max_retries (int): The maximum number of retries after a failing openai call.
        retry_delay (int): The delay between each retry.
        exponential_base (int): The base for exponential backoff calculation.
        exponential_delay (int): The initial delay for exponential backoff.
        exponential_cap (Optional[int]): The maximum delay for exponential backoff.
        jitter (bool): Whether to add jitter to the delay (to avoid concurrent firing).
        debug_stream_function_call (bool): Prints the arguments streamed by OpenAI during function call when streaming enabled.
    """

    def __init__(
        self,
        leading_prompts: Optional[List[Dict]] = None,
        trailing_prompts: Optional[List[Dict]] = None,
        max_consecutive_function_calls: int = MAX_CONSECUTIVE_CALLS,
        optimize_context: bool = False,
        message_history_size: Optional[int] = None,
        max_retries=15,
        retry_delay=10,
        exponential_base: int = 2,
        exponential_delay: int = 1,
        exponential_cap: Optional[int] = None,
        jitter: bool = True,
        debug_stream_function_call=False,
    ):
        self.leading_prompts = leading_prompts
        self.trailing_prompts = trailing_prompts
        self.max_consecutive_function_calls = max_consecutive_function_calls
        self.optimize_context = optimize_context
        self.message_history_size = message_history_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_base = exponential_base
        self.exponential_delay = exponential_delay
        self.exponential_cap = exponential_cap
        self.jitter = jitter
        self.debug_stream_function_call = debug_stream_function_call


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

    def to_openai_function_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_jsonapi_schema(),
            },
        }

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


class JAImsFuncWrapper:
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
        function: Optional[Callable[..., Any]],
        function_tool_descriptor: JAImsFunctionToolDescriptor,
    ):
        self.function = function
        self.function_tool = function_tool_descriptor

    def call(self, params: Dict[str, Any]) -> Any:
        """
        Calls the wrapped function with the passed parameters if the function is not None.
        Returns None otherwise.

        Parameters
        ----------
            params : dict
                the parameters passed to the wrapped function
        """
        return self.function(**params) if self.function else None


class JAImsFunctionToolResponse:
    """
    This class offers a way to interact with the agent to trigger events after a tool is called.
    It can be used as a response from a tool call.
    It is not mandatory, return this class from your function tools if you want to interact with the agent to alter the flow of the execution.

    Attributes
    ----------
        content : Any
            the content of the response to be sent to the LLM
        stop: bool
            Whether the tool call should stop the current execution or not, defaults to False.
            This is meant to be used when the tool calling is set to "auto" and it is necessary to
            stop the current execution but, in case of parallel tool calling, all the other tools should be called regardless.
            The net result is that the result of each tool call won't be sent back to the LLM and not tracked in the history.
        halt: bool
            Whether the tool call should stop the current execution or not, defaults to False.
            This is meant to be used when the tool calling is set to "auto" and it is necessary to stop the current execution abruptly.
            This means that in the context of parallel tool calling, as soon as the halt is set to True, all subsequent tool calls are not executed and the current run will terminate.
            The net result is that the result of each tool call won't be sent back to the LLM and not tracked in the history.
        override_kwargs: JAImsOpenaiKWArgs (optional)
            The kwargs to be used to override the current kwargs when giving tool results back to the LLM.
            If parallel tools are called in the same iteration and more than one sets an override_kwargs, the last override_kwargs will be used since, by design, the results are sent back to the LLM in batch.
            Useful for instance to update the model version, the token size or the temperature.
        override_options: JAImsOptions (optional)
            The options to be used to override the current options when giving tool results back to the LLM.
            If parallel tools are called in the same iteration and each sets an override_options, the last override_options will be used since, by design, the results are sent back to the LLM in batch.
            Useful for instance to update the static leading and trailing prompts, finetune the max consecutive calls allowed and so on.
    """

    def __init__(
        self,
        content: Any,
        halt: bool = False,
        override_kwargs: Optional[JAImsOpenaiKWArgs] = None,
        override_options: Optional[JAImsOptions] = None,
    ):
        self.content = content
        self.halt = halt
        self.override_kwargs = override_kwargs
        self.override_options = override_options


class JAImsToolResults:
    """
    Passed by the tool handler delegate to the agent to push the results of the tool calls back to the LLM or to stop the current execution.

    Attributes
    ----------
        function_result_messages: List[Any]
            the list of function tool result messages to be sent to the LLM
        stop: bool
            Wether the agent should stop the current execution or not, defaults to False.
        override_kwargs: JAImsOpenaiKWArgs (optional)
            Kwargs to be used to override the current kwargs when giving tool results back to the LLM.
        override_options: JAImsOptions (optional)
            Options to be used to override the current options when giving tool results back to the LLM.
    """

    def __init__(
        self,
        function_result_messages: List[Any],
        stop: bool = False,
        override_kwargs: Optional[JAImsOpenaiKWArgs] = None,
        override_options: Optional[JAImsOptions] = None,
    ):
        self.function_result_messages = function_result_messages
        self.stop = stop
        self.override_kwargs = override_kwargs
        self.override_options = override_options


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
