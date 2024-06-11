import json
from typing import Any, Dict, List
import unittest
from enum import Enum

from pydantic import BaseModel, Field
from jaims import (
    JAImsFunctionToolDescriptor,
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

    def test_wrapped_function_is_callable(self):

        class MockClass(BaseModel):
            a: int = Field(description="Mock a")
            b: int = Field(description="Mock b")

        @jaimsfunctiontool(description="mock wrapped function")
        def mock_wrapped_function(
            input: str, number: int, flag: bool, items: List[str], mock_class: MockClass
        ):
            items_str = " ".join(items)
            return f"Hello {input} {number} {flag} {items_str} {mock_class.a} {mock_class.b}"

        payload = {
            "input": "Mondo",
            "number": 42,
            "flag": True,
            "items": ["a", "b"],
            "mock_class": {"a": 11, "b": 22},
        }

        result = mock_wrapped_function(payload)
        self.assertEqual(result, "Hello Mondo 42 True a b 11 22")

    # def test_decorator_accepts_empty_parameters(self):

    #     @jaimsfunctiontool(description="mock wrapped function")
    #     def mock_wrapped_function():
    #         return "Hello World"

    #     self.assertCountEqual(
    #         mock_wrapped_function.function_tool.params_descriptors, []
    #     )

    # def test_decorator_builds_primitive_parameters_without_descriptor(self):

    #     @jaimsfunctiontool(description="Add two numbers")
    #     def add_numbers(a: int, b: float, c: str, d: bool, e: None):
    #         return "hello test"

    #     expected_func_tool_descriptor = JAImsFunctionToolDescriptor(
    #         name="add_numbers",
    #         description="Add two numbers",
    #         params_descriptors=[
    #             JAImsParamDescriptor(
    #                 name="a", json_type=JAImsJsonSchemaType.NUMBER, description=""
    #             ),
    #             JAImsParamDescriptor(
    #                 name="b", json_type=JAImsJsonSchemaType.NUMBER, description=""
    #             ),
    #             JAImsParamDescriptor(
    #                 name="c", json_type=JAImsJsonSchemaType.STRING, description=""
    #             ),
    #             JAImsParamDescriptor(
    #                 name="d", json_type=JAImsJsonSchemaType.BOOLEAN, description=""
    #             ),
    #             JAImsParamDescriptor(
    #                 name="e", json_type=JAImsJsonSchemaType.NULL, description=""
    #             ),
    #         ],
    #     )

    #     self.assertEqual(add_numbers.function_tool, expected_func_tool_descriptor)

    # def test_decorator_builds_structured_parameters_without_descriptor(self):

    #     class MockAddress:
    #         def __init__(self, street: str, city: str):
    #             self.street = street
    #             self.city = city

    #     class Gender(Enum):
    #         MALE = "male"
    #         FEMALE = "female"
    #         OTHER = "other"

    #     class NumberEnum(Enum):
    #         ONE = 1
    #         TWO = 2
    #         THREE = 3

    #     class MockPerson:
    #         def __init__(
    #             self,
    #             name: str,
    #             gender: Gender,
    #             magic_number: NumberEnum,
    #             addresses: List[MockAddress],
    #             funny_list: list,
    #             metadata: Dict,
    #         ):
    #             self.name = name
    #             self.gender = gender
    #             self.addresses = addresses
    #             self.funny_list = funny_list
    #             self.magic_number = magic_number
    #             self.metadata = metadata

    #     expected_func_tool_descriptor = JAImsFunctionToolDescriptor(
    #         name="store_person",
    #         description="store person info",
    #         params_descriptors=[
    #             JAImsParamDescriptor(
    #                 name="id", json_type=JAImsJsonSchemaType.STRING, description=""
    #             ),
    #             JAImsParamDescriptor(
    #                 name="person",
    #                 json_type=JAImsJsonSchemaType.OBJECT,
    #                 description="",
    #                 attributes_params_descriptors=[
    #                     JAImsParamDescriptor(
    #                         name="name",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="",
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="gender",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="",
    #                         enum_values=["male", "female", "other"],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="magic_number",
    #                         json_type=JAImsJsonSchemaType.NUMBER,
    #                         description="",
    #                         enum_values=[1, 2, 3],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="addresses",
    #                         json_type=JAImsJsonSchemaType.ARRAY,
    #                         description="",
    #                         array_type_descriptors=[
    #                             JAImsParamDescriptor(
    #                                 name="MockAddress",
    #                                 json_type=JAImsJsonSchemaType.OBJECT,
    #                                 description="",
    #                                 attributes_params_descriptors=[
    #                                     JAImsParamDescriptor(
    #                                         name="street",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                     JAImsParamDescriptor(
    #                                         name="city",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                 ],
    #                             )
    #                         ],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="funny_list",
    #                         json_type=JAImsJsonSchemaType.ARRAY,
    #                         description="",
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="metadata",
    #                         json_type=JAImsJsonSchemaType.OBJECT,
    #                         description="",
    #                     ),
    #                 ],
    #             ),
    #         ],
    #     )

    #     @jaimsfunctiontool(description="store person info")
    #     def store_person(id: str, person: MockPerson):
    #         return "hello test"

    #     result_tool_descriptor = store_person.function_tool
    #     if result_tool_descriptor != expected_func_tool_descriptor:
    #         print("\n\nRESULT:")
    #         print(json.dumps(result_tool_descriptor.to_dict(), indent=4))
    #         print("\n\nEXPECTED:")
    #         print(json.dumps(expected_func_tool_descriptor.to_dict(), indent=4))

    #     self.assertEqual(result_tool_descriptor, expected_func_tool_descriptor)

    # def test_decorator_builds_structured_parameters_with_descriptor_strings(self):

    #     class MockAddress:
    #         def __init__(self, street: str, city: str):
    #             self.street = street
    #             self.city = city

    #     class Gender(Enum):
    #         MALE = "male"
    #         FEMALE = "female"
    #         OTHER = "other"

    #     class MockPerson:
    #         def __init__(
    #             self,
    #             name: str,
    #             gender: Gender,
    #             addresses: List[MockAddress],
    #         ):
    #             self.name = name
    #             self.gender = gender
    #             self.addresses = addresses

    #     expected_func_tool_descriptor = JAImsFunctionToolDescriptor(
    #         name="store_person",
    #         description="store person info",
    #         params_descriptors=[
    #             JAImsParamDescriptor(
    #                 name="id",
    #                 json_type=JAImsJsonSchemaType.STRING,
    #                 description="the identifier of the person",
    #             ),
    #             JAImsParamDescriptor(
    #                 name="person",
    #                 json_type=JAImsJsonSchemaType.OBJECT,
    #                 description="a person record",
    #                 attributes_params_descriptors=[
    #                     JAImsParamDescriptor(
    #                         name="name",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="",
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="gender",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="",
    #                         enum_values=["male", "female", "other"],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="addresses",
    #                         json_type=JAImsJsonSchemaType.ARRAY,
    #                         description="",
    #                         array_type_descriptors=[
    #                             JAImsParamDescriptor(
    #                                 name="MockAddress",
    #                                 json_type=JAImsJsonSchemaType.OBJECT,
    #                                 description="",
    #                                 attributes_params_descriptors=[
    #                                     JAImsParamDescriptor(
    #                                         name="street",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                     JAImsParamDescriptor(
    #                                         name="city",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                 ],
    #                             )
    #                         ],
    #                     ),
    #                 ],
    #             ),
    #         ],
    #     )

    #     @jaimsfunctiontool(
    #         description="store person info",
    #         param_descriptors={
    #             "id": "the identifier of the person",
    #             "person": "a person record",
    #         },
    #     )
    #     def store_person(id: str, person: MockPerson):
    #         return "hello test"

    #     result_tool_descriptor = store_person.function_tool

    #     self.assertEqual(result_tool_descriptor, expected_func_tool_descriptor)

    # def test_decorator_builds_structured_parameters_with_descriptor_strings_and_dicts(
    #     self,
    # ):

    #     class MockAddress:
    #         def __init__(self, street: str, city: str):
    #             self.street = street
    #             self.city = city

    #     class Gender(Enum):
    #         MALE = "male"
    #         FEMALE = "female"
    #         OTHER = "other"

    #     class MockPerson:
    #         def __init__(
    #             self,
    #             name: str,
    #             gender: Gender,
    #             addresses: List[MockAddress],
    #         ):
    #             self.name = name
    #             self.gender = gender
    #             self.addresses = addresses

    #     expected_func_tool_descriptor = JAImsFunctionToolDescriptor(
    #         name="store_person",
    #         description="store person info",
    #         params_descriptors=[
    #             JAImsParamDescriptor(
    #                 name="id",
    #                 json_type=JAImsJsonSchemaType.STRING,
    #                 description="the identifier of the person",
    #             ),
    #             JAImsParamDescriptor(
    #                 name="person",
    #                 json_type=JAImsJsonSchemaType.OBJECT,
    #                 description="a person record",
    #                 attributes_params_descriptors=[
    #                     JAImsParamDescriptor(
    #                         name="name",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="name of the person",
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="gender",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="gender of the person",
    #                         enum_values=["male", "female", "other"],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="addresses",
    #                         json_type=JAImsJsonSchemaType.ARRAY,
    #                         description="list of addresses",
    #                         array_type_descriptors=[
    #                             JAImsParamDescriptor(
    #                                 name="MockAddress",
    #                                 json_type=JAImsJsonSchemaType.OBJECT,
    #                                 description="an address record",
    #                                 attributes_params_descriptors=[
    #                                     JAImsParamDescriptor(
    #                                         name="street",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="street name",
    #                                     ),
    #                                     JAImsParamDescriptor(
    #                                         name="city",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="city name",
    #                                     ),
    #                                 ],
    #                             )
    #                         ],
    #                     ),
    #                 ],
    #             ),
    #         ],
    #     )

    #     @jaimsfunctiontool(
    #         description="store person info",
    #         param_descriptors={
    #             "id": "the identifier of the person",
    #             "person": {
    #                 "description": "a person record",
    #                 "attributes": {
    #                     "name": "name of the person",
    #                     "gender": "gender of the person",
    #                     "addresses": {
    #                         "description": "list of addresses",
    #                         "array_types": {
    #                             "MockAddress": {
    #                                 "description": "an address record",
    #                                 "attributes": {
    #                                     "street": "street name",
    #                                     "city": "city name",
    #                                 },
    #                             },
    #                         },
    #                     },
    #                 },
    #             },
    #         },
    #     )
    #     def store_person(id: str, person: MockPerson):
    #         return "hello test"

    #     result_tool_descriptor = store_person.function_tool

    #     if result_tool_descriptor != expected_func_tool_descriptor:
    #         print("\n\nRESULT:")
    #         print(json.dumps(result_tool_descriptor.to_dict(), indent=4))
    #         print("\n\nEXPECTED:")
    #         print(json.dumps(expected_func_tool_descriptor.to_dict(), indent=4))

    #     self.assertEqual(result_tool_descriptor, expected_func_tool_descriptor)

    # def test_decorator_builds_structured_parameters_with_mixed_descriptor_strings_and_dicts(
    #     self,
    # ):

    #     class MockAddress:
    #         def __init__(self, street: str, city: str):
    #             self.street = street
    #             self.city = city

    #     class Gender(Enum):
    #         MALE = "male"
    #         FEMALE = "female"
    #         OTHER = "other"

    #     class MockPerson:
    #         def __init__(
    #             self,
    #             name: str,
    #             gender: Gender,
    #             addresses: List[MockAddress],
    #         ):
    #             self.name = name
    #             self.gender = gender
    #             self.addresses = addresses

    #     expected_func_tool_descriptor = JAImsFunctionToolDescriptor(
    #         name="store_person",
    #         description="store person info",
    #         params_descriptors=[
    #             JAImsParamDescriptor(
    #                 name="id",
    #                 json_type=JAImsJsonSchemaType.STRING,
    #                 description="the identifier of the person",
    #             ),
    #             JAImsParamDescriptor(
    #                 name="person",
    #                 json_type=JAImsJsonSchemaType.OBJECT,
    #                 description="a person record",
    #                 attributes_params_descriptors=[
    #                     JAImsParamDescriptor(
    #                         name="name",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="name of the person",
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="gender",
    #                         json_type=JAImsJsonSchemaType.STRING,
    #                         description="gender of the person",
    #                         enum_values=["male", "female", "other"],
    #                     ),
    #                     JAImsParamDescriptor(
    #                         name="addresses",
    #                         json_type=JAImsJsonSchemaType.ARRAY,
    #                         description="list of addresses",
    #                         array_type_descriptors=[
    #                             JAImsParamDescriptor(
    #                                 name="MockAddress",
    #                                 json_type=JAImsJsonSchemaType.OBJECT,
    #                                 description="",
    #                                 attributes_params_descriptors=[
    #                                     JAImsParamDescriptor(
    #                                         name="street",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                     JAImsParamDescriptor(
    #                                         name="city",
    #                                         json_type=JAImsJsonSchemaType.STRING,
    #                                         description="",
    #                                     ),
    #                                 ],
    #                             )
    #                         ],
    #                     ),
    #                 ],
    #             ),
    #         ],
    #     )

    #     @jaimsfunctiontool(
    #         description="store person info",
    #         param_descriptors={
    #             "id": "the identifier of the person",
    #             "person": {
    #                 "description": "a person record",
    #                 "attributes": {
    #                     "name": "name of the person",
    #                     "gender": {
    #                         "json_type": "string",
    #                         "description": "gender of the person",
    #                         "enum": ["male", "female", "other"],
    #                     },
    #                     "addresses": "list of addresses",
    #                 },
    #             },
    #         },
    #     )
    #     def store_person(id: str, person: MockPerson):
    #         return "hello test"

    #     result_tool_descriptor = store_person.function_tool

    #     if result_tool_descriptor != expected_func_tool_descriptor:
    #         print("\n\nRESULT:")
    #         print(json.dumps(result_tool_descriptor.to_dict(), indent=4))
    #         print("\n\nEXPECTED:")
    #         print(json.dumps(expected_func_tool_descriptor.to_dict(), indent=4))

    #     self.assertEqual(result_tool_descriptor, expected_func_tool_descriptor)


if __name__ == "__main__":
    unittest.main()
