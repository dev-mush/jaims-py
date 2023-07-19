from core import (
    JAImsAgent,
    JAImsFuncWrapper,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsGPTModel,
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

    agent = JAImsAgent(
        functions=[sum_func_wrapper, result_func_wrapper],
        model=JAImsGPTModel.GPT_3_5_TURBO_16K,
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.send_messages(
            [{"role": "user", "content": user_input}],
            stream=stream,
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
