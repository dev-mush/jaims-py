from core import (
    JAImsAgent,
    JAImsFuncWrapper,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    GPTModel,
)


def sum(a: int, b: int):
    print("----performing sum----")
    print(a, b)
    print("----------------------")
    return a + b


def main():
    stream = False

    func_wrapper = JAImsFuncWrapper(
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

    agent = JAImsAgent(
        functions=[func_wrapper],
        model=GPTModel.GPT_3_5_TURBO_16K,
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


if __name__ == "__main__":
    main()
