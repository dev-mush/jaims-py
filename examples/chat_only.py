import json
from jaims import (
    JAImsAgent,
    JAImsFuncWrapper,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsOpenaiKWArgs,
    BaseGPTModel, JAImsGPTModel,
    JAImsTransactionStorageInterface,
)


class MockTransactionStorage(JAImsTransactionStorageInterface):
    def store_transaction(self, request: dict, response: dict):
        print("----storing transaction----")
        print(json.dumps(request, indent=4))
        print("---------------------------")
        print(json.dumps(response, indent=4))
        print("---------------------------")


def main():
    stream = False

    agent = JAImsAgent(
        openai_kwargs=JAImsOpenaiKWArgs(
            model=BaseGPTModel(
                "mistralai/Mistral-7B-Instruct-v0.1", 32768, 0, 0, None,
            ),
            stream=stream,
            tools=[],
        ),
        openai_api_key="EMPTY",
        openai_base_url="http://34.124.189.144:8000/v1",
        transaction_storage=MockTransactionStorage(),
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.run(
            [{"role": "user", "content": user_input}],
        )

        if response:
            if stream:
                for chunk in response:
                    print(chunk, end="", flush=True)
                print("\n")

            else:
                print(response)

    expenses = agent.get_expenses()
    for expense in expenses:
        print(expense)


if __name__ == "__main__":
    main()
