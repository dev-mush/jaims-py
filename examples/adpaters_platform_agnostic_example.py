from jaims import (
    JAImsAgent,
    JAImsFunctionTool,
    JAImsDefaultHistoryManager,
    JAImsFunctionToolDescriptor,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsMessage,
    JAImsLLMConfig,
)


def sum(a: int, b: int):
    print("----performing sum----")
    print(a, b)
    print("----------------------")
    return a + b


def store_sum(result: int):
    print("----storing sum----")
    print(result)
    print("-------------------")


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


def main():
    stream = True
    model = "gpt-4o"  # use JAImsModelCode.GPT_4o to avoid typing / remembering the model name
    # model = "gemini-1.5-pro-latest"
    provider = "openai"  # either "openai" or "google"
    # provider = "google"

    agent = JAImsAgent.build(
        model=model,
        provider=provider,
        config=JAImsLLMConfig(
            temperature=0.5,
            max_tokens=2000,
        ),
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
            result_func_wrapper,
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
