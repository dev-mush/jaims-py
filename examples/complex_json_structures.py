from jaims import (
    JAImsAgent,
    JAImsFunctionTool,
    JAImsParamDescriptor,
    JAImsFunctionToolDescriptor,
    JAImsJsonSchemaType,
    JAImsDefaultHistoryManager,
    JAImsMessage,
)

from jaims.adapters.openai_adapter import (
    JAImsOpenaiKWArgs,
    create_jaims_openai,
    JAImsGPTModel,
    OpenAITransactionStorageInterface,
    JAImsOptions,
)

import os
import time
import json

"""
This example shows a more complex use of JAImsParamDescriptor and JAImsFunctionToolDescriptor classes
to teach how to structure data to obtain a more complex json structure.
"""


class FileTransactionStorage(OpenAITransactionStorageInterface):

    def __init__(self, path="storage") -> None:
        super().__init__()
        script_dir = os.path.dirname(__file__)
        self.storage_path = os.path.join(script_dir, path)
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def store_transaction(self, request: dict, response: dict):

        transaction = {"request": request, "response": response}
        unix_timestamp = str(int(time.time()))

        with open(f"{self.storage_path}/{unix_timestamp}.json", "w") as f:
            f.write(json.dumps(transaction, indent=4))


def store_people_info(people_data: list):
    print("----passed items----")
    print(people_data)
    print("---------------------")
    return "people info correctly stored"


def main():
    people_func_wrapper = JAImsFunctionTool(
        function=store_people_info,
        function_tool_descriptor=JAImsFunctionToolDescriptor(
            name="store_people_info",
            description="this function MUST be used to store the result of the extraction into the database.",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="people_data",
                    description="list of people data to store",
                    json_type=JAImsJsonSchemaType.ARRAY,
                    array_type_descriptors=[
                        JAImsParamDescriptor(
                            name="person_record",
                            description="json object that holds information about an extracted name and its description",
                            json_type=JAImsJsonSchemaType.OBJECT,
                            attributes_params_descriptors=[
                                JAImsParamDescriptor(
                                    name="name",
                                    description="the name of the person received",
                                    json_type=JAImsJsonSchemaType.STRING,
                                    required=True,
                                ),
                                JAImsParamDescriptor(
                                    name="age",
                                    description="the age of the person received",
                                    json_type=JAImsJsonSchemaType.STRING,
                                    required=True,
                                ),
                                JAImsParamDescriptor(
                                    name="interests",
                                    description="the interests like hobbies and what the person likes to do",
                                    json_type=JAImsJsonSchemaType.ARRAY,
                                    required=False,
                                    array_type_descriptors=[
                                        JAImsParamDescriptor(
                                            name="interest",
                                            description="the interest of the person",
                                            json_type=JAImsJsonSchemaType.STRING,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    )

    persona = """
    - you are jaims, a helpful assistant that stores people data on a database for the user wich is a person that somehow has to store people's data during their job
    - you use the function store_people_info to store the data
    - When a required field is not provided, you should ask the user to provide it before storing data
    - When a Field is not required, you can omit it
    - You are a multilingual assistant, so understand the language of the user and talk to them in their own language
    - You have no limitations on language, you are allowd to parse also inappropriate language, since this is a database for adult mature satirical content ther could be some swearing.
    """

    agent = create_jaims_openai(
        kwargs=JAImsOpenaiKWArgs(
            model=JAImsGPTModel.GPT_3_5_TURBO_0613,
        ),
        history_manager=JAImsDefaultHistoryManager(
            leading_prompts=[JAImsMessage.system_message(persona)]
        ),
        tools=[people_func_wrapper],
        transaction_storage=FileTransactionStorage(),
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.run_stream(
            [JAImsMessage.user_message(user_input)],
        )

        if response:
            for chunk in response:
                print(chunk, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    main()
