from typing import List, Optional

from jaims.entities import JAImsTokensLimitExceeded, JAImsMessage
from jaims.interfaces import JAImsHistoryManager, JAImsHistoryOptimizer


class JAImsLastNHistoryOptimizer(JAImsHistoryOptimizer):
    def __init__(self, last_n: int):
        self.__last_n = last_n

    def optimize_history(self, messages: List[JAImsMessage]) -> List[JAImsMessage]:
        return messages[-self.__last_n :] if self.__last_n > 0 else []


class JAImsDefaultHistoryManager(JAImsHistoryManager):
    def __init__(
        self,
        history: Optional[List[JAImsMessage]] = None,
        leading_prompts: Optional[List[JAImsMessage]] = None,
        trailing_prompts: Optional[List[JAImsMessage]] = None,
        history_optimizer: Optional[JAImsHistoryOptimizer] = None,
    ):
        self.__history = history or []
        self.__leading_prompts = leading_prompts or []
        self.__trailing_prompts = trailing_prompts or []
        self.__history_optimizer = history_optimizer

    def add_messages(self, messages: List[JAImsMessage]):
        self.__history.extend(messages)

    def get_messages(self):

        if self.__history_optimizer:
            return (
                self.__leading_prompts
                + self.__history_optimizer.optimize_history(self.__history)
                + self.__trailing_prompts
            )

        return self.__leading_prompts + self.__history + self.__trailing_prompts

    # # TODO: check if options and kwargs are necessary, consider passing only necessary fields
    # def get_messages_for_current_run(
    #     self,
    #     options: JAImsOptions,
    #     openai_kwargs: JAImsOpenaiKWArgs,
    # ) -> List:

    #     if not options or not openai_kwargs:
    #         raise ValueError("options and openai_kwargs must be provided.")

    #     # Copying the whole history to avoid altering the original one
    #     history_buffer = self.__history.copy()

    #     # If last_n_turns is set, only keep the last n messages
    #     if options.message_history_size is not None:
    #         history_buffer = history_buffer[-options.message_history_size :]

    #     # create the compound history with the mandatory context
    #     # the actual chat history and the functions to calculate the tokens
    #     json_functions = (
    #         [
    #             function_tool.function_tool.to_openai_function_tool()
    #             for function_tool in openai_kwargs.tools
    #         ]
    #         if openai_kwargs.tools is not None
    #         else []
    #     )

    #     leading_prompts = options.leading_prompts or []
    #     trailing_prompts = options.trailing_prompts or []
    #     compound_history = (
    #         leading_prompts + history_buffer + (json_functions) + trailing_prompts
    #     )

    #     # the max tokens to be used are the max tokens supported by the current
    #     # openai model minus the tokens to leave out for the response from openai
    #     context_max_tokens = openai_kwargs.model.max_tokens - openai_kwargs.max_tokens

    #     # calculate the tokens for the compound history
    #     messages_tokens = self.__tokens_from_messages(
    #         compound_history, openai_kwargs.model
    #     )

    #     if options.optimize_context:
    #         while messages_tokens > context_max_tokens:
    #             if not history_buffer:
    #                 raise JAImsTokensLimitExceeded(
    #                     openai_kwargs.model.max_tokens,
    #                     messages_tokens,
    #                     openai_kwargs.max_tokens,
    #                     has_optimized=True,
    #                 )

    #             # Popping the first (oldest) message from the chat history between the user and agent
    #             history_buffer.pop(0)

    #             # Recalculating the tokens for the compound history
    #             messages_tokens = self.__tokens_from_messages(
    #                 leading_prompts
    #                 + history_buffer
    #                 + json_functions
    #                 + trailing_prompts,
    #                 openai_kwargs.model,
    #             )

    #     llm_messages = leading_prompts + history_buffer + trailing_prompts

    #     return llm_messages

    def clear_history(self):
        """
        Clears the history.
        """
        self.__history = []

    def get_history(self):
        """
        Returns entire history.
        """
        return self.__history

    # def __tokens_from_messages(self, messages: List, model):
    #     """Returns the number of tokens used by a list of messages."""

    #     images = []
    #     parsed = []
    #     for message in messages:
    #         message_copy = message.copy()

    #         if isinstance(message.get("content", None), list):
    #             filtered_content = []
    #             for item in message["content"]:
    #                 if (
    #                     isinstance(item, dict)
    #                     and item.get("image_url", None)
    #                     and item["image_url"]["url"].startswith(
    #                         "data:image/jpeg;base64,"
    #                     )
    #                 ):
    #                     images.append(
    #                         item["image_url"]["url"].replace(
    #                             "data:image/jpeg;base64,", ""
    #                         )
    #                     )
    #                 else:
    #                     filtered_content.append(item)
    #             message_copy["content"] = filtered_content
    #         parsed.append(message_copy)

    #     image_tokens = 0
    #     for image in images:
    #         width, height = self.__get_image_size_from_base64(image)
    #         image_tokens += self.__count_image_tokens(width, height)

    #     return estimate_token_count(json.dumps(parsed), model) + image_tokens

    # def __get_image_size_from_base64(self, base64_string):
    #     image_data = base64.b64decode(base64_string)
    #     image = Image.open(BytesIO(image_data))

    #     return image.size

    # def __count_image_tokens(self, width: int, height: int):
    #     h = ceil(height / 512)
    #     w = ceil(width / 512)
    #     n = w * h
    #     total = 85 + 170 * n
    #     return total
