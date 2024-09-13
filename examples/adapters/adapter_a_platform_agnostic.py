from jaims import (
    Agent,
    DefaultHistoryManager,
    Message,
    LLMParams,
    Config,
    jaimsfunctiontool,
)


@jaimsfunctiontool(
    description="use this function when the user wants to sum two numbers",
    params_descriptions={"a": "the first operand", "b": "the second operand"},
)
def sum(a: int, b: int):
    print("\n----performing sum----")
    print(a, b)
    print("----------------------")
    return a + b


@jaimsfunctiontool(
    description="use this function when the user wants to subtract two numbers",
    params_descriptions={"a": "the first operand", "b": "the second operand"},
)
def multiply(a: int, b: int):
    print("\n----performing multiplication----")
    print(a, b)
    print("----------------------------------")
    return a * b


@jaimsfunctiontool(
    description="use this function when the user wants to store the result of an operation",
)
def store_result(result: int):
    print("\n----storing result----")
    print(result)
    print("-------------------")


def main():
    stream = True
    model = "gemini-1.5-pro"
    provider = "vertex"

    agent = Agent.build(
        model=model,
        provider=provider,
        config=Config(
            platform_specific_options={
                "project_id": "your-project-id",
                "location": "europe-west1",
            }
        ),
        llm_params=LLMParams(
            temperature=0.5,
            max_tokens=2000,
        ),
        history_manager=DefaultHistoryManager(),
        tools=[
            sum,
            multiply,
            store_result,
        ],
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
