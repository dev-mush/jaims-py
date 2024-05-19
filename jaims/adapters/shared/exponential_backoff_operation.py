from enum import Enum
import random
from typing import Any, Callable
from ...entities import JAImsOptions, JAImsMaxRetriesExceeded
import time
import logging


class ErrorHandlingMethod(Enum):
    FAIL = "fail"
    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


def call_with_exponential_backoff(
    operation: Callable[[], Any],
    error_handler: Callable[[Exception], ErrorHandlingMethod],
    options: JAImsOptions,
) -> Any:
    retries = 0
    sleep_time = options.retry_delay
    backoff_time = options.exponential_delay
    logger = logging.getLogger(__name__)
    last_exception = None

    while retries <= options.max_retries:
        try:
            return operation()
        except Exception as error:
            last_exception = error
            logger.error(f"Client Error:\n{error}\n")
            error_handling_method = error_handler(error)

            if error_handling_method == ErrorHandlingMethod.FAIL:
                raise error

            if error_handling_method == ErrorHandlingMethod.RETRY:
                sleep_time = options.retry_delay

            elif error_handling_method == ErrorHandlingMethod.EXPONENTIAL_BACKOFF:
                logger.warning(f"Performing exponential backoff")
                jitter = 1 + options.jitter * random.random()
                backoff_time = backoff_time * options.exponential_base * jitter

                if (
                    options.exponential_cap is not None
                    and backoff_time > options.exponential_cap
                ):
                    backoff_time = options.exponential_cap * jitter

                sleep_time = backoff_time

            retries += 1
            logger.warning(
                f"Attempting {retries} of {options.max_retries}. in {sleep_time} seconds"
            )
            time.sleep(sleep_time)

    logger.error(f"Max retries exceeded! Operation failed {options.max_retries} times.")
    raise JAImsMaxRetriesExceeded(options.max_retries, last_exception)
