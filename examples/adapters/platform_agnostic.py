from jaims import (
    JAImsAgent,
    JAImsDefaultHistoryManager,
    JAImsMessage,
    JAImsLLMConfig,
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
    model = "claude-3-5-sonnet-20240620"  # use JAImsModelCode.GPT_4o to avoid typing / remembering the model name
    # model = "gemini-1.5-pro-latest"
    provider = "anthropic"  # either "openai" or "google"
    # provider = "google"

    agent = JAImsAgent.build(
        model=model,
        provider=provider,
        config=JAImsLLMConfig(
            temperature=0.5,
            max_tokens=2000,
        ),
        history_manager=JAImsDefaultHistoryManager(),
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
