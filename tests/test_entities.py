import unittest
from jaims.entities import JAImsParamDescriptor, JAImsJsonSchemaType


class TestJAImsParamDescriptor(unittest.TestCase):
    def test_param_descriptor(self):
        # Create a parameter descriptor
        param_descriptor = JAImsParamDescriptor(
            name="param_name",
            description="Parameter description",
            json_type=JAImsJsonSchemaType.STRING,
            required=True,
        )

        # Check the attributes of the parameter descriptor
        self.assertEqual(param_descriptor.name, "param_name")
        self.assertEqual(param_descriptor.description, "Parameter description")
        self.assertEqual(param_descriptor.json_type, JAImsJsonSchemaType.STRING)
        self.assertEqual(param_descriptor.required, True)

    def test_param_descriptor_eq(self):
        # Create two parameter descriptors with the same attributes
        param_descriptor1 = JAImsParamDescriptor(
            name="param_name",
            description="Parameter description",
            json_type=JAImsJsonSchemaType.OBJECT,
            attributes_params_descriptors=[
                JAImsParamDescriptor(
                    name="attribute_name",
                    description="Attribute description",
                    json_type=JAImsJsonSchemaType.STRING,
                    required=True,
                ),
                JAImsParamDescriptor(
                    name="array_attribute_name",
                    description="Array description",
                    json_type=JAImsJsonSchemaType.ARRAY,
                    required=True,
                    array_type_descriptors=[
                        JAImsParamDescriptor(
                            name="array_item_name",
                            description="Array item description",
                            json_type=JAImsJsonSchemaType.NUMBER,
                            required=True,
                        )
                    ],
                ),
            ],
            required=True,
        )

        param_descriptor2 = JAImsParamDescriptor(
            name="param_name",
            description="Parameter description",
            json_type=JAImsJsonSchemaType.OBJECT,
            attributes_params_descriptors=[
                JAImsParamDescriptor(
                    name="attribute_name",
                    description="Attribute description",
                    json_type=JAImsJsonSchemaType.STRING,
                    required=True,
                ),
                JAImsParamDescriptor(
                    name="array_attribute_name",
                    description="Array description",
                    json_type=JAImsJsonSchemaType.ARRAY,
                    required=True,
                    array_type_descriptors=[
                        JAImsParamDescriptor(
                            name="array_item_name",
                            description="Array item description",
                            json_type=JAImsJsonSchemaType.NUMBER,
                            required=True,
                        )
                    ],
                ),
            ],
            required=True,
        )

        # Check if the two parameter descriptors are equal
        self.assertEqual(param_descriptor1, param_descriptor2)

    def test_param_descriptor_jsonapi_schema(self):
        # Create a parameter descriptor with attributes
        attribute_descriptor = JAImsParamDescriptor(
            name="attribute_name",
            description="Attribute description",
            json_type=JAImsJsonSchemaType.STRING,
            required=True,
        )
        param_descriptor = JAImsParamDescriptor(
            name="param_name",
            description="Parameter description",
            json_type=JAImsJsonSchemaType.OBJECT,
            attributes_params_descriptors=[attribute_descriptor],
            required=True,
        )

        # Get the JSONAPI schema of the parameter descriptor
        schema = param_descriptor.get_json_schema()

        # Check the JSONAPI schema
        expected_schema = {
            "type": "object",
            "description": "Parameter description",
            "properties": {
                "attribute_name": {
                    "type": "string",
                    "description": "Attribute description",
                }
            },
            "required": ["attribute_name"],
        }
        self.assertEqual(schema, expected_schema)

    def test_param_descriptor_from_dictionary(self):

        expected_param = JAImsParamDescriptor(
            name="param_name",
            description="Parameter description",
            json_type=JAImsJsonSchemaType.OBJECT,
            attributes_params_descriptors=[
                JAImsParamDescriptor(
                    name="attribute_name",
                    description="Attribute description",
                    json_type=JAImsJsonSchemaType.STRING,
                    required=True,
                ),
                JAImsParamDescriptor(
                    name="array_attribute_name",
                    description="Array description",
                    json_type=JAImsJsonSchemaType.ARRAY,
                    required=True,
                    array_type_descriptors=[
                        JAImsParamDescriptor(
                            name="array_item_name",
                            description="Array item description",
                            json_type=JAImsJsonSchemaType.NUMBER,
                            required=True,
                        )
                    ],
                ),
            ],
            required=True,
        )

        param_descriptor_dict = {
            "name": "param_name",
            "description": "Parameter description",
            "json_type": "object",
            "required": True,
            "attributes_params_descriptors": [
                {
                    "name": "attribute_name",
                    "description": "Attribute description",
                    "json_type": "string",
                    "required": True,
                },
                {
                    "name": "array_attribute_name",
                    "description": "Array description",
                    "json_type": "array",
                    "required": True,
                    "array_type_descriptors": [
                        {
                            "name": "array_item_name",
                            "description": "Array item description",
                            "json_type": "number",
                            "required": True,
                        }
                    ],
                },
            ],
        }

        # Create a parameter descriptor from the dictionary
        param_descriptor = JAImsParamDescriptor.from_dict(param_descriptor_dict)

        self.assertEqual(param_descriptor, expected_param)


if __name__ == "__main__":
    unittest.main()
