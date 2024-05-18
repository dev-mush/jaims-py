import unittest
from unittest.mock import MagicMock
from jaims.adapters.shared.entities import JAImsMaxRetriesExceeded
from jaims.adapters.shared.exponential_backoff_operation import (
    call_with_exponential_backoff,
    ErrorHandlingMethod,
    JAImsOptions,
)
import time


class TestExponentialBackoffOperation(unittest.TestCase):
    def test_call_succeeds(self):
        # Define the operation to be called
        def operation():
            return "Success"

        # Define the error handler
        def error_handler(error):
            return ErrorHandlingMethod.FAIL

        # Define the options
        options = JAImsOptions(
            max_retries=3,
            retry_delay=1,
            exponential_delay=2,
            exponential_base=2,
            exponential_cap=10,
            jitter=False,
        )

        result = call_with_exponential_backoff(operation, error_handler, options)
        self.assertEqual(result, "Success")

    def test_retry_succeeds(self):
        times = 0

        def operation():
            nonlocal times
            times += 1
            if times == 2:
                return "Success"
            raise Exception("Some error")

        # Define the error handler
        def error_handler(error):
            return ErrorHandlingMethod.RETRY

        # Define the options
        options = JAImsOptions(
            max_retries=3,
            retry_delay=1,
            exponential_delay=2,
            exponential_base=2,
            exponential_cap=10,
            jitter=False,
        )

        result = call_with_exponential_backoff(operation, error_handler, options)
        self.assertEqual(result, "Success")

    def test_retry_fails_after_max_retries(self):
        count = 0

        def operation():
            nonlocal count
            count += 1
            raise Exception("Some error")

        def error_handler(error):
            return ErrorHandlingMethod.RETRY

        # Define the options
        options = JAImsOptions(
            max_retries=3,
            retry_delay=1,
            exponential_delay=2,
            exponential_base=1,
            exponential_cap=10,
            jitter=False,
        )

        # Call the function under test
        with self.assertRaises(JAImsMaxRetriesExceeded):
            call_with_exponential_backoff(operation, error_handler, options)

        self.assertEqual(count, 4)

    def test_exponential_backoff_performed(self):
        count = 0

        def operation():
            nonlocal count
            if count == 0:
                count += 1
                raise Exception("Some error")

            return "Success"

        # Define the error handler
        def error_handler(error):
            return ErrorHandlingMethod.EXPONENTIAL_BACKOFF

        # Define the options
        options = JAImsOptions(
            max_retries=3,
            retry_delay=1,
            exponential_delay=2,
            exponential_base=1,
            exponential_cap=10,
            jitter=False,
        )

        now = time.time()
        result = call_with_exponential_backoff(operation, error_handler, options)
        elapsed = time.time() - now
        self.assertEqual(result, "Success")

        # check the test lasted about 2 seconds
        self.assertAlmostEqual(elapsed, 2, delta=0.1)

    def test_exponential_backoff_fails_after_max_retries(self):
        count = 0

        def operation():
            nonlocal count
            count += 1
            raise Exception("Some error")

        # Define the error handler
        def error_handler(error):
            return ErrorHandlingMethod.EXPONENTIAL_BACKOFF

        # Define the options
        options = JAImsOptions(
            max_retries=2,
            retry_delay=1,
            exponential_delay=1,
            exponential_base=1,
            exponential_cap=10,
            jitter=False,
        )

        # Call the function under test
        with self.assertRaises(JAImsMaxRetriesExceeded):
            call_with_exponential_backoff(operation, error_handler, options)

        self.assertEqual(count, 3)

    def test_exponential_backoff_is_capped(self):
        count = 0

        def operation():
            nonlocal count
            if count == 0:
                count += 1
                raise Exception("Some error")
            return "Success"

        # Define the error handler
        def error_handler(error):
            return ErrorHandlingMethod.EXPONENTIAL_BACKOFF

        # Define the options

        options = JAImsOptions(
            max_retries=3,
            retry_delay=1,
            exponential_delay=2,
            exponential_base=10,
            exponential_cap=2,
            jitter=False,
        )

        now = time.time()
        result = call_with_exponential_backoff(operation, error_handler, options)
        elapsed = time.time() - now

        self.assertEqual(result, "Success")
        self.assertAlmostEqual(elapsed, 2, delta=0.1)


if __name__ == "__main__":
    unittest.main()
