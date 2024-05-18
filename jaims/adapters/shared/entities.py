from __future__ import annotations
from typing import Optional


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
