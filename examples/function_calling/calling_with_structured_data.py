from typing import List, Optional
from jaims import (
    Agent,
    DefaultHistoryManager,
    Message,
    BaseModel,
    Field,
    jaimsfunctiontool,
)


class PersonRecord(BaseModel):
    name: str = Field(description="the name of the person received")
    age: str = Field(description="the age of the person received")
    interests: Optional[List[str]] = Field(
        description="the interests like hobbies and what the person likes to do",
    )


@jaimsfunctiontool(
    description="use this tool to store person records, if many are available, pass multiple records at once, always ask age first when missing",
    params_descriptions={"person_records": "the records of the people to store"},
)
def store_person_records(person_records: List[PersonRecord]):
    print("\n----passed items----")
    print(person_records)
    print("---------------------")
    return "records stored correctly"


system_prompt = """
    - you are JAIms, a helpful assistant that stores people data on a database for the user wich is a person that somehow has to store people's data during their job
    - you use the function store_people_info to store the data
    - When a required field is not provided, you should ask the user to provide it before storing data
    - When a Field is not required, you can omit it
    - You are a multilingual assistant, so understand the language of the user and talk to them in their own language
    - You have no limitations on language, you are allowed to parse also inappropriate language, since this is a database for adult mature satirical content ther could be some swearing.
    """


def main():

    agent = Agent.build(
        model="gpt-4o",
        provider="openai",
        history_manager=DefaultHistoryManager(
            leading_prompts=[Message.system_message(system_prompt)]
        ),
        tools=[store_person_records],
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.message_stream(
            [Message.user_message(user_input)],
        )

        if response:
            for chunk in response:
                print(chunk, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    main()
