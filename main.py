from core import JAImsAgent, JAImsFuncWrapper, JAImsParamDescriptor, JsonSchemaType


def sum(a: int, b: int):
    return a + b


def main():
    stream = True

    func_wrapper = JAImsFuncWrapper(
        function=sum,
        name="sum",
        description="use it when the user wants you to sum two numbers",
        params_descriptors=[
            JAImsParamDescriptor(
                name="a",
                description="the first number to be summed",
                json_type=JsonSchemaType.NUMBER,
            ),
            JAImsParamDescriptor(
                name="b",
                description="the second number to be summed",
                json_type=JsonSchemaType.NUMBER,
            ),
        ],
    )

    agent = JAImsAgent(functions=[func_wrapper])

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.send_messages(
            [{"role": "user", "content": user_input}], stream=stream
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
