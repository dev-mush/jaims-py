import json
from jaims import (
    BaseModel,
    Field,
    FunctionToolDescriptor,
    Agent,
    Message,
)
from typing import List, Optional


class PersonRecord(BaseModel):
    name: str = Field(description="the name of a person")
    age: Optional[str] = Field(
        default=None, description="the age of the person received"
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="the interests like hobbies and what the person likes to do",
    )


class PeopleExtraction(BaseModel):
    people: List[PersonRecord] = Field(
        description="the people records stored in the database"
    )


extract_people = FunctionToolDescriptor(
    name="extract_people",
    description="use this tool to extract people records from the database",
    params=PeopleExtraction,
)


unstructured_text = """
In the quaint village of Maplewood, four friends often gathered at the old town café. Sarah, 34, an avid gardener, always shared stories about her latest floral arrangements. Next to her, you’d usually find Tom, who loved nothing more than hiking through the surrounding lush forests. On the opposite side sat Emily, 29, who enjoyed painting landscapes inspired by their outings. Lastly, there was Mark, known for his impressive collection of vintage vinyl records. Together, they made weekends lively with their diverse interests.
"""


if __name__ == "__main__":

    # run tool on agent instance
    agent = Agent.build(
        model="gpt-4-turbo",
        provider="openai",
    )

    people = agent.run_tool(
        extract_people, messages=[Message.user_message(unstructured_text)]
    )

    print(json.dumps(people.model_dump(), indent=4))

    # run tool on class
    people = Agent.run_tool_model(
        model="gpt-4-turbo",
        provider="openai",
        descriptor=extract_people,
        messages=[Message.user_message(unstructured_text)],
    )

    print(json.dumps(people.model_dump(), indent=4))
