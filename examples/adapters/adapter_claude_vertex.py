from jaims import (
    DefaultHistoryManager,
    Message,
    jaimsfunctiontool,
    Config,
)
from jaims.adapters.anthropic_adapter.adapter import (
    AnthropicAdapter,
    AnthropicParams,
)
from jaims.agent import Agent


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


def main():
    stream = True

    adapter = AnthropicAdapter(
        provider="vertex",
        params=AnthropicParams(model="claude-3-5-sonnet@20240620"),
        config=Config(
            platform_specific_options={
                "region": "europe-west1",
                "project_id": "your-project-id",
            }
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
