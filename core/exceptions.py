class TokensLimitExceeded(Exception):
    def __init__(self, max_tokens, messages_tokens, llm_buffer, has_optimized):
        message = f"Max tokens: {max_tokens}\n LLM Answer Buffer: {llm_buffer}\n Messages tokens: {messages_tokens}\n Messages Optimized: {has_optimized}  "
        super().__init__(message)


class MissingOpenaiAPIKeyException(Exception):
    def __init__(self):
        message = "Missing OPENAI_API_KEY, set environment variable OPENAI_API_KEY or pass it as a parameter to the agent constructor."
        super().__init__(message)


class OpenAIErrorException(Exception):
    def __init__(self, message, openai_error):
        super().__init__(message)
        self.openai_error = openai_error


class UnexpectedFunctionCall(Exception):
    def __init__(self, func_name):
        message = f"Unexpected function call: {func_name}"
        super().__init__(message)
