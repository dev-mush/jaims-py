from jaims import (
    JAImsFunctionTool,
    JAImsFunctionToolDescriptor,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsMessage,
    JAImsDefaultHistoryManager,
)

from jaims.adapters.google_generative_ai_adapter import (
    create_jaims_gemini,
)


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

    history_manager = JAImsDefaultHistoryManager(
        leading_prompts=[
            JAImsMessage.system_message(
                "You are JAIms, a helpful assistant that helps the user with math operations."
            )
        ],
    )

    agent = create_jaims_gemini(
        model="gemini-1.5-pro-latest",
        history_manager=history_manager,
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
