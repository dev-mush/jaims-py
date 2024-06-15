import unittest
from jaims import JAImsFunctionTool, JAImsFunctionToolDescriptor
from pydantic import BaseModel, Field


class TestJAImsFunctionTool(unittest.TestCase):

    def test_function_tool_calls_wrapped_function_with_base_model_from_raw(self):

        class MockClass(BaseModel):
            mock_str: str = Field(description="Mock string")
            mock_num: int = Field(description="Mock number")

        tool_descriptor = JAImsFunctionToolDescriptor(
            name="mock_tool", description="Mock tool", params=MockClass
        )

        result = False

        def mock_function(mock_param: MockClass):
            if mock_param.mock_str == "mock" and mock_param.mock_num == 1:
                nonlocal result
                result = True

        tool = JAImsFunctionTool(descriptor=tool_descriptor, function=mock_function)

        tool.call_raw(**{"mock_str": "mock", "mock_num": 1})

        self.assertTrue(result)

    def test_function_tool_calls_wrapped_function_with_no_typing_from_raw(self):

        class MockClass(BaseModel):
            mock_str: str = Field(description="Mock string")
            mock_num: int = Field(description="Mock number")

        tool_descriptor = JAImsFunctionToolDescriptor(
            name="mock_tool", description="Mock tool", params=MockClass
        )

        result = False

        def mock_function(args):
            if isinstance(args, MockClass):
                nonlocal result
                result = True

        tool = JAImsFunctionTool(descriptor=tool_descriptor, function=mock_function)

        tool.call_raw(**{"mock_str": "mock", "mock_num": 1})

        self.assertTrue(result)

    def test_function_tool_calls_instance_methods(self):

        class MockClass(BaseModel):
            mock_str: str = Field(description="Mock string")
            mock_num: int = Field(description="Mock number")

        tool_descriptor = JAImsFunctionToolDescriptor(
            name="mock_tool", description="Mock tool", params=MockClass
        )

        class MockClassInstance:

            def __init__(self):
                self.result = False

            def mock_method(self, mock_param: MockClass):
                if mock_param.mock_str == "mock" and mock_param.mock_num == 1:
                    self.result = True

        instance = MockClassInstance()

        tool = JAImsFunctionTool(
            descriptor=tool_descriptor, function=instance.mock_method
        )

        tool.call_raw(**{"mock_str": "mock", "mock_num": 1})

        self.assertTrue(instance.result)

    def test_function_tool_uses_formatter_to_provide_data_to_wrapped_function(self):

        class MockClass(BaseModel):
            mock_str: str = Field(description="Mock string")
            mock_num: int = Field(description="Mock number")

        tool_descriptor = JAImsFunctionToolDescriptor(
            name="mock_tool", description="Mock tool", params=MockClass
        )

        result = False

        def mock_function(first_param: str, second_param: int):
            if first_param == "mock" and second_param == 1:
                nonlocal result
                result = True

        def custom_formatter(data):
            return (), {
                "first_param": data["mock_str"],
                "second_param": data["mock_num"],
            }

        tool = JAImsFunctionTool(
            descriptor=tool_descriptor,
            function=mock_function,
            formatter=custom_formatter,
        )

        tool.call_raw(**{"mock_str": "mock", "mock_num": 1})

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
