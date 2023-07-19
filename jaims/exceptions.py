class JAImsTokensLimitExceeded(Exception):
    def __init__(self, max_tokens, messages_tokens, llm_buffer, has_optimized):
        message = f"Max tokens: {max_tokens}\n LLM Answer Buffer: {llm_buffer}\n Messages tokens: {messages_tokens}\n Messages Optimized: {has_optimized}  "
        super().__init__(message)


class JAImsMissingOpenaiAPIKeyException(Exception):
    def __init__(self):
        message = "Missing OPENAI_API_KEY, set environment variable OPENAI_API_KEY or pass it as a parameter to the agent constructor."
        super().__init__(message)


class JAImsOpenAIErrorException(Exception):
    def __init__(self, message, openai_error):
        super().__init__(message)
        self.openai_error = openai_error


class JAImsMaxConsecutiveFunctionCallsExceeded(Exception):
    def __init__(self, max_consecutive_calls):
        message = f"Max consecutive function calls exceeded: {max_consecutive_calls}"
        super().__init__(message)


class JAImsUnexpectedFunctionCall(Exception):
    def __init__(self, func_name):
        message = f"Unexpected function call: {func_name}"
        super().__init__(message)
