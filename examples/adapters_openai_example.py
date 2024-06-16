import json
import os
import time
from jaims import (
    JAImsFunctionTool,
    JAImsDefaultHistoryManager,
    JAImsFunctionToolDescriptor,
    JAImsMessage,
    jaimsfunctiontool,
)

from jaims.adapters.openai_adapter import (
    JAImsOpenaiKWArgs,
    create_jaims_openai,
    OpenAITransactionStorageInterface,
)


@jaimsfunctiontool(
    description="use this function when the user wants to sum two numbers",
    params_descriptions={"a": "the first operand", "b": "the second operand"},
)
def sum(a: int, b: int):
    print("----performing sum----")
    print(a, b)
    print("----------------------")
    return a + b


@jaimsfunctiontool(
    description="use this function when the user wants to multiply two numbers",
    params_descriptions={"a": "the first operand", "b": "the second operand"},
)
def multiply(a: int, b: int):
    print("----performing multiply----")
    print(a, b)
    print("----------------------")
    return a * b


@jaimsfunctiontool(
    description="use this function when the user wants to store the result of a sum",
    params_descriptions={"result": "the result of a sum"},
)
def store_sum(result: int):
    print("----storing sum----")
    print(result)
    print("-------------------")


@jaimsfunctiontool(
    description="use this function when the user wants to store the result of a multiply",
    params_descriptions={"result": "the result of a multiply"},
)
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
        tools=[sum, multiply, store_sum, store_multiply],
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break

        if stream:
            response = agent.message_stream(
                [JAImsMessage.user_message(text=user_input)],
            )
            for chunk in response:
                print(chunk, end="", flush=True)
            print("\n")
        else:
            response = agent.message(
                [JAImsMessage.user_message(text=user_input)],
            )
            print(response)


if __name__ == "__main__":
    main()
