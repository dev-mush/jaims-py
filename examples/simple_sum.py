from jaims import (
    JAImsAgent,
    JAImsFuncWrapper,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsOpenaiKWArgs,
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

    sum_func_wrapper = JAImsFuncWrapper(
        function=sum,
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
    )

    multiply_func_wrapper = JAImsFuncWrapper(
        function=multiply,
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
    )

    result_func_wrapper = JAImsFuncWrapper(
        function=store_sum,
        name="store_sum_result",
        description="this function MUST be called every time after a sum function is called to store its result.",
        params_descriptors=[
            JAImsParamDescriptor(
                name="result",
                description="the result of a sum",
                json_type=JAImsJsonSchemaType.NUMBER,
            ),
        ],
    )

    result_multiply_func_wrapper = JAImsFuncWrapper(
        function=store_multiply,
        name="store_multiply_result",
        description="this function MUST be called every time after a multiply function is called to store its result.",
        params_descriptors=[
            JAImsParamDescriptor(
                name="result",
                description="the result of a multiply",
                json_type=JAImsJsonSchemaType.NUMBER,
            ),
        ],
    )

    agent = JAImsAgent(
        openai_kwargs=JAImsOpenaiKWArgs(
            stream=stream,
            tools=[
                sum_func_wrapper,
                multiply_func_wrapper,
                result_func_wrapper,
                result_multiply_func_wrapper,
            ],
        )
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
