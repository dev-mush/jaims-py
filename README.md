<p align="center">
    <img width="300" src="https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9" >
</p>


# jAIms 

_My name is Bot, jAIMs Bot._ 🕶️

jAIms is a lightweight Python framework built on top of the OpenAI library that lets you create powerful LLM agents.
It is designed with simplicity and ease of use in mind and only depends on `openai` and `tiktoken`.

## Installation

```bash
pip install jaims-py
```

## 👨‍💻 Usage

Building an agent is as simple as this:

```python
from jaims import JAImsAgent

agent = JAImsAgent()

response = agent.run([
    {
        "role": "user",
        "content": "Hi!"
    }
])

print(response)
```

The parameters accepted by the `run` method are those specified in the [official OpenAI docs](https://platform.openai.com/docs/api-reference/chat/create).

### ⚙️ Functions

Of course, an agent is just a chatbot if it doesn't support functions. jAIms uses the built-in OpenAI function feature to call functions you pass to it. Here's an example where we create a simple sum function and make a simple agent that lets you sum two numbers:

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
    response = agent.run(
        [{"role": "user", "content": user_input}],
        stream=True,
    )

    for chunk in response:
        print(chunk, end="", flush=True)
        
        print("\n")
```

### ✨ Other features

- Complete control over openai call parameters (temperature, top_p, n, max_tokens, etc.)
- Automatic chat history management
- Configuration of the OpenAI model to use
- Injectable prompt to shape agent behavior
- Safety checks to prevent the agent from endlessly looping over function calls

I will routinely update the examples to demonstrate more advanced features.
Also, I've made sure to document the code as best as I can; everything should be self-explanatory; I plan to add a proper documentation in the future if this project gets enough traction.

## 🤖 Supported models

Currently, jAIms supports the new OpenAI models with functions enabled, specifically:

- `gpt-3.5-turbo-0613`
- `gpt-3.5-turbo-16k-0613`
- `gpt-4-0613`

I'm not planning to add support for non-OpenAI models at the moment, but contributions are always appreciated.

## ⚠️ Project status

This is a work in progress. I still need to write some tests and add many features, but the core functionality is there.
I'm creating this framework because I need a lightweight and easy-to-use framework to create LLM agents. This project may not be as advanced as tools like [langchain](https://github.com/hwchase17/langchain) and others, but if you need a simple tool to create agents based on the OpenAI API, you might find jAIms useful.


TODOS:

- [ ] Add tests
- [ ] Add more examples
- [ ] Add more chat history optimization strategies
- [ ] Add function calling callbacks 
- [ ] Add history persistance

## 📝 License

The license will be MIT, but I need to add this properly.
