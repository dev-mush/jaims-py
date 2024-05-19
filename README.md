<p align="center">
    <img width="300" src="https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9" >
</p>


# JAIms 

_My name is Bot, JAIms Bot._ 🕶️

JAIms is a lightweight Python package that lets you build powerful LLM-Based agents with ease. It is platform agnostic, so you can focus on integrating AI into your software and let JAIms handle the boilerplate of communicating with the LLM API. 
JAIms natively supports OpenAI's GPT models and Google's gemini models (based on google's generative ai), and it can be easily extended to connect to your own model and endpoints.

## Installation

To avoid overcluttering your project with dependencies, by running:

```bash
pip install jaims-py
```

You will get the core package that is provider independent (meaning, it won't install any dependencies other than Pillow). In order to also install the built in providers you can run:

```bash
pip install jaims-py[openai,google]
```

## 👨‍💻 Usage

Building an agent is as simple as this:

```python
from jaims import JAImsAgent, JAImsMessage

agent = JAImsAgent.build(
    model="gpt-4o",
    provider="openai",
)

response = agent.run([JAImsMessage.user_message("Hello, how are you?")])

print(response)
```

### ⚙️ Functions

Of course, an agent is just a chatbot if it doesn't support functions. JAIms uses the built-in OpenAI tools feature to call the functions you pass to it. Here's an example where we create a simple sum function and make a simple agent that lets you sum two numbers:

```python
def sum(a: int, b: int):
    return a + b

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

def main():
    agent = JAImsAgent.build(
        model="gpt-4o",
        provider="openai",
        config=JAImsLLMConfig(
            temperature=1.0,
        ),
        history_manager=JAImsDefaultHistoryManager(
            history=[
                JAImsMessage.assistant_message(
                    text="Hello, I am JAIms, your personal assistant, How can I help you today?"
                )
            ]
        ),
        tools=[
            sum_func_wrapper,
        ],
    )

    while True:
        user_input = input("> ")
        if user_input == "exit":
            break

       
        response = agent.run_stream(
            [JAImsMessage.user_message(text=user_input)],
        )
        for chunk in response:
            print(chunk, end="", flush=True)
        print("\n")
```

Check outh the examples folder for more advanced use cases.

### ✨ Main Features

- Lightweight (like...really, I'm rather obsessed with this, so dependencies are kept to a minimum)
- Built in support for OpenAI and Google's gemini models
- Built in history manager to allow fast creation of chatbots, this can be easily extended to support more advanced history management strategies.
- Support for images for vision models 🖼️
- Error handling and exponential backoff for built in providers (openai, google)

I will routinely update the examples to demonstrate more advanced features.

## ⚠️ Project status

This is a work in progress. I still need to write some (many) tests and add more QoL features, but the core functionality is there.
I'm creating this package because I need a lightweight and easy-to-use framework to create LLM agents / connect to foundational LLM providers with ease. The guiding philosophy behind is to build a platform agnostic interface to easily integrate software with foundational llm models to leverage AI features, making sure that I can easily switch between LLM providers without needing to change my code, yet I'm doing my best to ensure provider specific features and extensibility.

I'm currently using and maintaining this package for my own projects and those of the company I work for, but have opted for a open source by default approach to allow others to benefit from it and force myself to keep the code clean and well documented.


TODOS:

- [ ] Simplify function passing
- [ ] Add tests
- [ ] Refactor documentation and logging entirely   


## 📝 License

The license will be MIT, but I need to add this properly.
