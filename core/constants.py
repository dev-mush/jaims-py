from enum import Enum

DEFAULT_MAX_TOKENS = 512
MAX_SUBSEQUENT_CALLS = 5


class GPTModel(Enum):
    """
    The OPENAI GPT models available.
    Only those that support functions are listed, so just:
    gpt-3.5-turbo-0613, gpt-3-5-turbo-16k-0613, gpt-4-0613
    """

    GPT_3_5_TURBO = ("gpt-3.5-turbo-0613", 4096)
    GPT_3_5_TURBO_16K = ("gpt-3.5-turbo-16k-0613", 16384)
    GPT_4 = ("gpt-4-0613", 8192)

    def __init__(self, string, max_tokens):
        self.string = string
        self.max_tokens = max_tokens

    def __str__(self):
        return self.string
