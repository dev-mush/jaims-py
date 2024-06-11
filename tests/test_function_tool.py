import unittest
from jaims import JAImsFunctionTool, JAImsFunctionToolDescriptor
from pydantic import BaseModel, Field


class TestJAImsFunctionTool(unittest.TestCase):

    def test_function_tool_calls_wrapped_function_with_base_model(self):

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

        tool({"mock_str": "mock", "mock_num": 1})

        self.assertTrue(result)

    def test_function_tool_calls_wrapped_function_that_expects_kwargs(self):

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

        tool({"mock_str": "mock", "mock_num": 1})

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
