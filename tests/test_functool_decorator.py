from typing import List, Optional
import unittest

from pydantic import BaseModel, Field
from jaims import (
    jaimsfunctiontool,
)


class TestJAIMSFunctionToolDecorator(unittest.TestCase):

    def test_decorator_sets_passed_tool_name(self):

        @jaimsfunctiontool(name="add_numbers_alt", description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.descriptor.name, "add_numbers_alt")

    def test_decorator_sets_passed_description(self):

        @jaimsfunctiontool(name="add_numbers", description="Add two numbers alt")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.descriptor.description, "Add two numbers alt")

    def test_decorator_sets_function_name_when_not_passed(self):

        @jaimsfunctiontool(description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add_numbers.descriptor.name, "add_numbers")

    def test_decorated_function_gets_called_by_wrapper(self):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")
            b: int = Field(description="Mock b")

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(
            input: str,
            number: int,
            flag: bool,
            items: List[str],
            mock_class: MockClass,
        ) -> str:
            items_str = " ".join(items)
            return f"Hello {input} {number} {flag} {items_str} {mock_class.a} {mock_class.b}"

        result = mock_wrapped_function(
            "Mondo", 42, True, ["a", "b"], MockClass(a=11, b=22)
        )
        self.assertEqual(result, "Hello Mondo 42 True a b 11 22")

    def test_decorated_function_gets_called_with_formatter(self):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")
            b: int = Field(description="Mock b")

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(
            input: str,
            number: int,
            flag: bool,
            items: List[str],
            mock_class: MockClass,
        ) -> str:
            items_str = " ".join(items)
            return f"Hello {input} {number} {flag} {items_str} {mock_class.a} {mock_class.b}"

        payload = {
            "input": "Mondo",
            "number": 42,
            "flag": True,
            "items": ["a", "b"],
            "mock_class": {"a": 11, "b": 22},
        }

        result = mock_wrapped_function.call_raw(**payload)
        self.assertEqual(result, "Hello Mondo 42 True a b 11 22")

    def test_decorated_tool_descriptor_param_type_is_base_model_when_passed_base_model_only(
        self,
    ):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")
            b: int = Field(description="Mock b")

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(mock_class: MockClass):
            return f"{mock_class.a} {mock_class.b}"

        self.assertIsInstance(mock_wrapped_function.descriptor.params, type(MockClass))

    def test_decorator_skips_return_type_in_json_schema(self):

        @jaimsfunctiontool(description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        output_schema = add_numbers.descriptor.json_schema()

        self.assertNotIn("return", output_schema["properties"])

    def test_decorator_applies_descriptions_when_passed(self):

        @jaimsfunctiontool(
            name="add_numbers",
            description="Add two numbers",
            params_descriptions={"a": "First number", "b": "Second number"},
        )
        def add_numbers(a: int, b: Optional[int] = None) -> int:
            return a + (b or 1)

        output_schema = add_numbers.descriptor.json_schema()

        self.assertEqual(
            output_schema["properties"]["a"]["description"], "First number"
        )
        self.assertEqual(
            output_schema["properties"]["b"]["description"], "Second number"
        )

    def test_decorator_marks_fields_with_default_values_as_not_required(self):

        @jaimsfunctiontool(description="Add two numbers")
        def add_numbers(a: int, b: Optional[int] = None, c: int = 4) -> int:
            return a + (b or 1)

        output_schema = add_numbers.descriptor.json_schema()

        self.assertNotIn("b", output_schema["required"])
        self.assertNotIn("c", output_schema["required"])

    def test_decorator_works_on_instances(self):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")

        class MockClassMethods:
            @jaimsfunctiontool(description="mock wrapped function")
            def mock_wrapped_function(
                self, param_one: str, param_two: MockClass
            ) -> str:
                self.called = True
                return f"{param_one} {param_two.a}"

        mock_instance = MockClassMethods()
        payload = {"param_one": "Hello", "param_two": {"a": 42}}
        result = mock_instance.mock_wrapped_function.call_raw(**payload)
        self.assertTrue(mock_instance.called)
        self.assertEqual(result, "Hello 42")

    def test_decorator_works_on_functions(self):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(param_one: str, param_two: MockClass) -> str:
            return f"{param_one} {param_two.a}"

        payload = {"param_one": "Hello", "param_two": {"a": 42}}
        result = mock_wrapped_function.call_raw(**payload)
        self.assertEqual(result, "Hello 42")


if __name__ == "__main__":
    unittest.main()
