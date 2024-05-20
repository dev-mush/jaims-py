import unittest
from jaims import (
    JAImsJsonSchemaType,
    JAImsFunctionToolDescriptor,
    JAImsParamDescriptor,
    jaimsfunctiontool,
)


class TestJAIMSFunctionToolDecorator(unittest.TestCase):

    def test_decorator_sets_passed_tool_name(self):

        @jaimsfunctiontool(name="add_numbers_alt", description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.function_tool.name, "add_numbers_alt")

    def test_decorator_sets_passed_description(self):

        @jaimsfunctiontool(name="add_numbers", description="Add two numbers alt")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.function_tool.description, "Add two numbers alt")

    def test_decorator_sets_function_name_when_not_passed(self):

        @jaimsfunctiontool(description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.function_tool.name, "add_numbers")

    def test_wrapped_function_is_callable(self):

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(input: str):
            return f"Hello {input}"

        payload = {"input": "Mondo"}

        result_one = mock_wrapped_function("World")
        result_two = mock_wrapped_function(**payload)
        self.assertEqual(result_one, "Hello World")
        self.assertEqual(result_two, "Hello Mondo")

    def test_decorator_accepts_empty_parameters(self):

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function():
            return "Hello World"

        self.assertCountEqual(
            mock_wrapped_function.function_tool.params_descriptors, []
        )

    def test_decorator_builds_parameters_without_descriptor(self):

        @jaimsfunctiontool(description="Add two numbers")
        def add_numbers(a: int, b: float, c: str, d: bool, e: None):
            return "hello test"

        expected_params = [
            JAImsParamDescriptor(
                name="a", json_type=JAImsJsonSchemaType.NUMBER, description=""
            ),
            JAImsParamDescriptor(
                name="b", json_type=JAImsJsonSchemaType.NUMBER, description=""
            ),
            JAImsParamDescriptor(
                name="c", json_type=JAImsJsonSchemaType.STRING, description=""
            ),
            JAImsParamDescriptor(
                name="d", json_type=JAImsJsonSchemaType.BOOLEAN, description=""
            ),
            JAImsParamDescriptor(
                name="e", json_type=JAImsJsonSchemaType.NULL, description=""
            ),
        ]

        params = add_numbers.function_tool.params_descriptors
        self.assertEqual(add_numbers.function_tool.params_descriptors, expected_params)


if __name__ == "__main__":
    unittest.main()
