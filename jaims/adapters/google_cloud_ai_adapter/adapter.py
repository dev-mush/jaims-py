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

from typing import List, Generator, Optional, Union
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
from vertexai.preview.generative_models import ToolConfig
from vertexai.preview import generative_models as preview_generative_models


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
    ):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=self.project_id, location=self.location)

    def call(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> JAImsMessage:
        vertex_messages = self.__jaims_messages_to_vertex(messages)
        vertex_tools = self.__jaims_tools_to_vertex(tools)
        vertex_kw_args = self.kwargs.copy_with_overrides(
            messages=vertex_messages,
            tools=vertex_tools,
            stream=False,
        )
        response = self.___get_openai_response(vertex_kw_args, self.options)
        assert isinstance(response, ChatCompletion)
        if self.transaction_storage:
            self.transaction_storage.store_transaction(
                request=vertex_kw_args.to_dict(),
                response=response.model_dump(exclude_none=True),
            )

        return self.__openai_chat_completion_to_jaims_message(response)

    def call_streaming(
        self, messages: List[JAImsMessage], tools: List[JAImsFunctionTool]
    ) -> Generator[JAImsStreamingMessage, None, None]:
        openai_messages = self.__jaims_messages_to_vertex(messages)
        openai_tools = self.__jaims_tools_to_vertex(tools)
        openai_kw_args = self.kwargs.copy_with_overrides(
            messages=openai_messages,
            tools=openai_tools,
            stream=True,
        )
        response = self.___get_openai_response(openai_kw_args, self.options)
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
                request=openai_kw_args.to_dict(),
                response=accumulated_delta.model_dump(exclude_none=True),
            )

    def __jaims_messages_to_vertex(self, messages: List[JAImsMessage]) -> List[dict]:

        def format_contents(contents: List[JAImsMessageContent]):
            if len(contents) == 1 and contents[0].type == JAImsContentTypes.TEXT:
                return contents[0].content
            else:
                raw_contents = []
                for c in contents:

                    if c.type == JAImsContentTypes.IMAGE:
                        raw_contents.append(
                            {"type": "image_url", "image_url": c.content}
                        )
                    elif c.type == JAImsContentTypes.TEXT:
                        raw_contents.append({"type": "text", "content": c.content})
                    else:
                        raise Exception(f"Unsupported content type: {c.type}")

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

        raw_messages = []
        for m in messages:
            if m.tool_responses:
                for tr in m.tool_responses:
                    raw_messages.append(
                        {
                            "role": "tool",
                            "name": tr.tool_name,
                            "tool_call_id": tr.tool_call_id,
                            "content": json.dumps(tr.response),
                        }
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

    def __jaims_tools_to_vertex(self, tools: List[JAImsFunctionTool]) -> List[dict]:
        raw_tools = []
        for t in tools:
            tool_raw_dict = {
                "type": "function",
                "function": {
                    "name": t.function_tool.name,
                    "description": t.function_tool.description,
                    "parameters": t.function_tool.get_jsonapi_schema(),
                },
            }
            raw_tools.append(tool_raw_dict)

        return raw_tools

    def __openai_chat_completion_to_jaims_message(
        self, completion: ChatCompletion
    ) -> JAImsMessage:
        if len(completion.choices) == 0:
            raise Exception("OpenAI returned an empty response.")

        message = completion.choices[0].message
        role = JAImsMessageRole(message.role)
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                JAImsToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    tool_args=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        content = None
        text = None
        if message.content:
            text = message.content
            content = [JAImsMessageContent(type=JAImsContentTypes.TEXT, content=text)]

        return JAImsMessage(
            role=role,
            contents=content,
            tool_calls=tool_calls,
            text=text,
            raw=message,
        )

    def __openai_chat_completion_choice_delta_to_jaims_message(
        self, accumulated_choice_delta: ChoiceDelta, current_chunk: ChatCompletionChunk
    ) -> JAImsStreamingMessage:

        current_choice = current_chunk.choices[0]
        role = (
            JAImsMessageRole(current_choice.delta.role)
            if current_choice.delta.role
            else accumulated_choice_delta.role
        )
        textDelta = current_choice.delta.content
        text = None
        contents = None
        function_tool_calls = None

        role = JAImsMessageRole(accumulated_choice_delta.role)
        if accumulated_choice_delta.content:
            text = accumulated_choice_delta.content
            contents = [
                JAImsMessageContent(
                    type=JAImsContentTypes.TEXT,
                    content=accumulated_choice_delta.content,
                )
            ]

        if current_choice.finish_reason and accumulated_choice_delta.tool_calls:
            function_tool_calls = []
            for tc in accumulated_choice_delta.tool_calls:
                if tc.function:
                    function_tool_calls.append(
                        JAImsToolCall(
                            id=tc.id or "",
                            tool_name=tc.function.name or "",
                            tool_args=(
                                json.loads(tc.function.arguments)
                                if tc.function.arguments
                                else None
                            ),
                        )
                    )

        return JAImsStreamingMessage(
            message=JAImsMessage(
                role=role,
                contents=contents,
                tool_calls=function_tool_calls,
                text=text,
                raw=accumulated_choice_delta,
            ),
            textDelta=textDelta,
        )

    def __handle_openai_error(self, error: openai.OpenAIError) -> ErrorHandlingMethod:
        # errors are handled according to the guidelines here: https://platform.openai.com/docs/guides/error-codes/api-errors (dated 03/10/2023)
        # this map indexes all the error that require a retry or an exponential backoff, every other error is a fail
        error_handling_map = {
            openai.RateLimitError: ErrorHandlingMethod.EXPONENTIAL_BACKOFF,
            openai.InternalServerError: ErrorHandlingMethod.RETRY,
            openai.APITimeoutError: ErrorHandlingMethod.RETRY,
        }

        for error_type, error_handling_method in error_handling_map.items():
            if isinstance(error, error_type):
                return error_handling_method

        return ErrorHandlingMethod.FAIL

    def ___get_openai_response(
        self,
        openai_kw_args: JAImsOpenaiKWArgs,
        call_options: JAImsOptions,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        retries = 0
        logger = logging.getLogger(__name__)
        # keeps how long to sleep between retries
        sleep_time = call_options.retry_delay
        # keeps track of the exponential backoff
        backoff_time = call_options.exponential_delay

        while retries < call_options.max_retries:
            try:
                client = OpenAI(api_key=self.api_key)
                kwargs = openai_kw_args.to_dict()
                response = client.chat.completions.create(
                    **kwargs,
                )

                return response
            except openai.OpenAIError as error:
                logger.error(f"OpenAI API error:\n{error}\n")
                error_handling_method = self.__handle_openai_error(error)

                if error_handling_method == ErrorHandlingMethod.FAIL:
                    raise Exception(f"OpenAI API error: {error}")

                if error_handling_method == ErrorHandlingMethod.RETRY:
                    sleep_time = call_options.retry_delay

                elif error_handling_method == ErrorHandlingMethod.EXPONENTIAL_BACKOFF:
                    logger.info(f"Performing exponential backoff")
                    jitter = 1 + call_options.jitter * random.random()
                    backoff_time = backoff_time * call_options.exponential_base * jitter

                    if (
                        call_options.exponential_cap is not None
                        and backoff_time > call_options.exponential_cap
                    ):
                        backoff_time = call_options.exponential_cap * jitter

                    sleep_time = backoff_time

                logger.warning(f"Retrying in {sleep_time} seconds")
                time.sleep(sleep_time)
                retries += 1

        max_retries_error = f"Max retries exceeded! OpenAI API call failed {call_options.max_retries} times."
        logger.error(max_retries_error)
        raise Exception(max_retries_error)

    def __merge_tool_calls(
        self,
        existing_tool_calls: Optional[List[ChoiceDeltaToolCall]],
        new_tool_calls_delta: List[ChoiceDeltaToolCall],
    ):
        if not existing_tool_calls:
            return new_tool_calls_delta

        new_tool_calls = existing_tool_calls[:]
        for new_call_delta in new_tool_calls_delta:
            # check the tall call is already being streamed
            existing_call = next(
                (item for item in new_tool_calls if item.index == new_call_delta.index),
                None,
            )
            # new tool call, add it to the list
            if not existing_call:
                new_tool_calls.append(new_call_delta)

            # existing tool call, update it
            else:
                # update tool type
                if (
                    existing_call.type != new_call_delta.type
                    and new_call_delta.type is not None
                ):
                    existing_call.type = new_call_delta.type

                # update tool id
                if (
                    existing_call.id != new_call_delta.id
                    and new_call_delta.id is not None
                ):
                    existing_call.id = new_call_delta.id

                # update function
                if new_call_delta.function:
                    if existing_call.function is None:
                        existing_call.function = new_call_delta.function
                    else:
                        # update function name
                        if (
                            existing_call.function.name != new_call_delta.function.name
                            and new_call_delta.function.name is not None
                        ):
                            existing_call.function.name = new_call_delta.function.name

                        # update function args
                        existing_call.function.arguments = (
                            existing_call.function.arguments or ""
                        ) + (new_call_delta.function.arguments or "")

        return new_tool_calls

    def __accumulate_choice_delta(
        self, accumulator: Optional[ChoiceDelta], new_delta: ChoiceDelta
    ) -> ChoiceDelta:
        if accumulator is None:
            return new_delta

        if new_delta.content:
            accumulator.content = (accumulator.content or "") + new_delta.content
        if new_delta.role:
            accumulator.role = new_delta.role
        if new_delta.tool_calls:
            accumulator.tool_calls = self.__merge_tool_calls(
                accumulator.tool_calls, new_delta.tool_calls
            )

        return accumulator


def create_jaims_openai(
    api_key: Optional[str] = None,
    options: Optional[JAImsOptions] = None,
    kwargs: Optional[JAImsOpenaiKWArgs] = None,
    transaction_storage: Optional[OpenAITransactionStorageInterface] = None,
    history_manager: Optional[JAImsHistoryManager] = None,
    tool_manager: Optional[JAImsToolManager] = None,
    tools: Optional[List[JAImsFunctionTool]] = None,
) -> JAImsAgent:
    """
    Creates a JAIms instance with an OpenAI adapter.

    Args:
        api_key (Optional[str], optional): The OpenAI API key. Defaults to None.
        options (Optional[JAImsOptions], optional): The options for the adapter. Defaults to None.
        kwargs (Optional[JAImsOpenaiKWArgs], optional): The keyword arguments for the adapter. Defaults to None.
        transaction_storage (Optional[JAImsTransactionStorageInterface], optional): The transaction storage interface. Defaults to None.
        history_manager (Optional[JAImsHistoryManager], optional): The history manager. Defaults to None.
        tool_manager (Optional[JAImsToolManager], optional): The tool manager. Defaults to None.
        tools (Optional[List[JAImsFunctionTool]], optional): The list of function tools. Defaults to None.

    Returns:
        JAImsAgent: The JAIms agent, initialized with the OpenAI adapter.
    """
    adapter = JAImsOpenaiAdapter(
        api_key=api_key,
        options=options,
        kwargs=kwargs,
        transaction_storage=transaction_storage,
    )

    agent = JAImsAgent(
        llm_interface=adapter,
        history_manager=history_manager,
        tool_manager=tool_manager,
        tools=tools,
    )

    return agent
