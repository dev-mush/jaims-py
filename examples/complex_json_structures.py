from jaims import (
    JAImsAgent,
    JAImsFuncWrapper,
    JAImsParamDescriptor,
    JAImsJsonSchemaType,
    JAImsGPTModel,
)


def store_people_info(people_data: list):
    print("----getting items----")
    print(people_data)
    print("---------------------")
    return list


def main():
    stream = True

    people_func_wrapper = JAImsFuncWrapper(
        function=store_people_info,
        name="store_people_info",
        description="this function MUST be used to store the result of the extraction into the database.",
        params_descriptors=[
            JAImsParamDescriptor(
                name="people_data",
                description="list of people data to store",
                json_type=JAImsJsonSchemaType.ARRAY,
                array_type_descriptors=[
                    JAImsParamDescriptor(
                        name="person_record",
                        description="json object that holds information about an extracted name and its description",
                        json_type=JAImsJsonSchemaType.OBJECT,
                        attributes_params_descriptors=[
                            JAImsParamDescriptor(
                                name="name",
                                description="the name of the person received",
                                json_type=JAImsJsonSchemaType.STRING,
                            ),
                            JAImsParamDescriptor(
                                name="age",
                                description="the age of the person received",
                                json_type=JAImsJsonSchemaType.STRING,
                                required=False,
                            ),
                            JAImsParamDescriptor(
                                name="interests",
                                description="the interests like hobbies and what the person likes to do",
                                json_type=JAImsJsonSchemaType.ARRAY,
                                required=False,
                                array_type_descriptors=[
                                    JAImsParamDescriptor(
                                        name="interest",
                                        description="the interest of the person",
                                        json_type=JAImsJsonSchemaType.STRING,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    agent = JAImsAgent(
        functions=[people_func_wrapper],
        model=JAImsGPTModel.GPT_3_5_TURBO_16K,
        initial_prompts=[
            {
                "role": "system",
                "content": "Only use the functions you have been provided with.",
            },
            {
                "role": "system",
                "content": "When a Field is not required, you can omit it.",
            },
            {
                "role": "system",
                "content": "When the user asks to store a list of people and some info about them, format them and use the function to store them",
            },
        ],
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.run(
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
