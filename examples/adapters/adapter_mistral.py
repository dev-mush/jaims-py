import json
import os
import time
from jaims import (
    Agent,
    DefaultHistoryManager,
    Message,
    jaimsfunctiontool,
)


from jaims.adapters.mistral_adapter import (
    JAImsMistralKWArgs,
    MistralTransactionStorageInterface,
)


from jaims.adapters.mistral_adapter import JAImsMistralAdapter


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


class FileTransactionStorage(MistralTransactionStorageInterface):
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

    adapter = JAImsMistralAdapter(
        kwargs=JAImsMistralKWArgs(
            model="mistral-large-latest",
            stream=stream,
        ),
    )

    agent = Agent(
        llm_interface=adapter,
        history_manager=DefaultHistoryManager(),
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
                [Message.user_message(text=user_input)],
            )
            for chunk in response:
                print(chunk, end="", flush=True)
            print("\n")
        else:
            response = agent.message(
                [Message.user_message(text=user_input)],
            )
            print(response)


if __name__ == "__main__":
    main()
