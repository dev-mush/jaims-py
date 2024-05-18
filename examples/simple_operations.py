import json
import os
import time
from jaims import (
    JAImsAgent,
    JAImsFunctionTool,
    JAImsDefaultHistoryManager,
    JAImsFunctionToolDescriptor,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsMessage,
)

from jaims.adapters.openai_adapter import (
    JAImsOpenaiKWArgs,
    create_jaims_openai,
    OpenAITransactionStorageInterface,
)

"""
This example shows how to perform parallel tool calls and how to use the 
JAImsTransactionStorageInterface to store each transaction with the LLM.
"""


def sum(a: int, b: int):
    print("----performing sum----")
    print(a, b)
    print("----------------------")
    return a + b


def multiply(a: int, b: int):
    print("----performing multiply----")
    print(a, b)
    print("----------------------")
    return a * b


def store_sum(result: int):
    print("----storing sum----")
    print(result)
    print("-------------------")


def store_multiply(result: int):
    print("----storing multiply----")
    print(result)
    print("-------------------")


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


def main():
    stream = True

    sum_func_wrapper = JAImsFunctionTool(
        function=sum,
        function_tool_descriptor=JAImsFunctionToolDescriptor(
            name="sum",
            description="use this function when the user wants to sum two numbers",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="a",
                    description="first operand",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
                JAImsParamDescriptor(
                    name="b",
                    description="second operand",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
            ],
        ),
    )

    multiply_func_wrapper = JAImsFunctionTool(
        function=multiply,
        function_tool_descriptor=JAImsFunctionToolDescriptor(
            name="multiply",
            description="use this function when the user wants to multiply two numbers",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="a",
                    description="first operand",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
                JAImsParamDescriptor(
                    name="b",
                    description="second operand",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
            ],
        ),
    )

    result_func_wrapper = JAImsFunctionTool(
        function=store_sum,
        function_tool_descriptor=JAImsFunctionToolDescriptor(
            name="store_sum_result",
            description="this function MUST be called every time after a sum function is called to store its result.",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="result",
                    description="the result of a sum",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
            ],
        ),
    )

    result_multiply_func_wrapper = JAImsFunctionTool(
        function=store_multiply,
        function_tool_descriptor=JAImsFunctionToolDescriptor(
            name="store_multiply_result",
            description="this function MUST be called every time after a multiply function is called to store its result.",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="result",
                    description="the result of a multiply",
                    json_type=JAImsJsonSchemaType.NUMBER,
                ),
            ],
        ),
    )

    agent = create_jaims_openai(
        kwargs=JAImsOpenaiKWArgs(
            model="gpt-4-turbo-2024-04-09",
            stream=stream,
        ),
        transaction_storage=FileTransactionStorage(),
        history_manager=JAImsDefaultHistoryManager(
            history=[
                JAImsMessage.assistant_message(
                    text="Hello, I am JAIms, your personal assistant."
                ),
                JAImsMessage.assistant_message(text="How can I help you today?"),
            ]
        ),
        tools=[
            sum_func_wrapper,
            multiply_func_wrapper,
            result_func_wrapper,
            result_multiply_func_wrapper,
        ],
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break

        if stream:
            response = agent.run_stream(
                [JAImsMessage.user_message(text=user_input)],
            )
            for chunk in response:
                print(chunk, end="", flush=True)
            print("\n")
        else:
            response = agent.run(
                [JAImsMessage.user_message(text=user_input)],
            )
            print(response)


if __name__ == "__main__":
    main()
