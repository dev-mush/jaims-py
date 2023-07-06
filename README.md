# 🕵️‍♂️ jAIms 

jAIms is a lightweight python framework built on top of the openai library that lets you create powerful llm agents.
It is designed with simplicity and ease of use in mind, it only depends on `openai` and `tiktoken`.

## ⚠️ Project status

This is a work in progress. I need to write some tests and add many features, but the core functionality is there.
I'm creating this framework because I need a lightweight and easy to use framework to create llm agents, this project pales in comparison with more advanced tools like [langchain](https://github.com/hwchase17/langchain) and the likes, but if you need a simple tool to create agents based on the openai api, you might find jAIms useful.

## Installation

## 👨‍💻 Usage

Building an agent is as simple as that:

```python
from jaims import JAImsAgent

agent = JAImsAgent()

response = agent.send_messages([
    {
        "role": "user",
        "content": "Hi!"
    }
])

print(response)
```

The messages accepted by the `send_messages` function are those specified in the [official openai docs](https://platform.openai.com/docs/api-reference/chat/create).

### ⚙️ Functions

Of course an agent is just a chatbot if there is no support for functions. jAIms uses the built in openai function feature to call functions you pass to it. Here's an example where we create a simple sum function and make a simple agent that lets you sum two numbers:

```python
import jaims


def sum(a: int, b: int):
    return a + b

# this is a class that wraps your function, it will 
# receive the actual function plus all the info required 
# by the llm to invoke it.
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

# instantiate the agent passing the functions
agent = JAImsAgent(
    functions=[func_wrapper],
    model=JAImsGPTModel.GPT_3_5_TURBO_16K,
)

# a simple loop that simulates a chatbot
while True:
    user_input = input("> ")
    if user_input == "exit":
        break
    response = agent.send_messages(
        [{"role": "user", "content": user_input}],
        stream=True,
    )

    for chunk in response:
        print(chunk, end="", flush=True)
        
        print("\n")
```

I will routinely update the examples to show more advanced features. 
Also I've made sure to document the code as better as I can, anything should be self explanatory.

## 🤖 Supported models

Right now jaims supports the new openai models with functions enabled, specifically:

- `gpt-3.5-turbo-0613`
- `gpt-3.5-turbo-16k-0613`
- `gpt-4-0613`

I'm not planning to add support for non-openai models right now, but any contribution is appreciated.

## 📝 License

should be MIT, but need to add this properly.
