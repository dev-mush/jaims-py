<p align="center">
    <img width="300" src="https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9" >
</p>


# JAIms 

_My name is Bot, JAIms Bot._ 🕶️

JAIms is a lightweight Python package that lets you build powerful LLM-Based agents or LLM powered applications with ease. It is platform agnostic, so you can focus on integrating AI into your software and let JAIms handle the boilerplate of communicating with the LLM API. 
The main goal of JAIms is to provide a simple and easy-to-use interface to leverage the power of LLMs in your software, without having to worry about the specifics of the underlying provider, and to seamlessly integrate LLM functionality with your own codebase.
JAIms natively supports OpenAI's GPT models and Google's gemini models, and it can be easily extended to connect to your own model and endpoints.

## Installation

To avoid overcluttering your project with dependencies, by running:

```bash
pip install jaims-py
```

You will get the core package that is provider independent (meaning, it won't install any dependencies other than Pillow and Pydantic). In order to also install the built in providers (currently openai and google) you can run:

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

### ⚙️ Function Tools

Of course, an agent is just a chatbot if it doesn't support functions. JAIms leverages LLMs function calling features seamlessly integrating with your python code.
It can both invoke your python functions, or use a platform agnostic tool descriptor to return formatted results that are easily consumed by your code (using pydantic models).

#### Function Invocation

```python
from jaims import JAImsAgent, JAImsMessage, jaimsfunctiontool

@jaimsfunctiontool()
def sum(a: int, b: int):
    print("invoked sum function")
    return a + b

agent = JAImsAgent.build(
    model="gpt-4o",
    provider="openai",
    tools=[sum],
)

response = agent.message([JAImsMessage.user_message("What is the sum of 42 and 420?")])
print(response)
```

#### Formatted Results

```python
import jaims
from typing import Optional


class MotivationalQuote(jaims.BaseModel):
    quote: str = jaims.Field(description="a motivational quote")
    author: Optional[str] = jaims.Field(
        default=None, description="the author of the quote, omit if it's your own"
    )


tool_descriptor = jaims.JAImsFunctionToolDescriptor(
    name="store_motivational_quote",
    description="use this tool to store a random motivational quote based on user's preferences",
    params=MotivationalQuote,
)


random_quote = jaims.JAImsAgent.run_tool_model(
    model="gpt-4o",
    provider="openai",
    descriptor=tool_descriptor,
    messages=[
        jaims.JAImsMessage.user_message("Motivate me in becoming a morning person.")
    ],
)
print(f"Quote: {random_quote.quote}\nAuthor: {random_quote.author or 'By an AI Poet'}")
```

But there is much more, check outh the examples folder for more advanced or nuanced use cases.

### ✨ Main Features

- Built in support for OpenAI and Google's gemini models (more coming soon).
- Function calling support even in streamed conversations with built in providers (openai, google).
- Built in conversation history management to allow fast creation of chatbots, this can be easily extended to support more advanced history management strategies.
- Image support for multimodal LLMs 🖼️
- Error handling and exponential backoff for built in providers (openai, google)

### 🧠 Guiding Principles

JAIms comes out of the necessity for a lightweight and easy-to-use framework to create LLM agents or integrate LLM functionality in python projects. Given the increasing work with both foundational and open source LLMs, JAIms has been designed as an abstraction layer to streamline fast creation of agentic business logic and seamless codebase integration.

In case you like to contribute, please keep in mind that I try to keep the code:

- **Modular**: any component is provided with a default basic implementation and an interface that can be easily extended for more complex use cases.
- **Type Hinted and Explicit**: I've done my best to type hint everything and document the codebase as much as possible to avoid digging into the code.
- **Tested**: Well...Let's just say I could have done better, but am planning to improve code coverage and test automation in the near future.
- **Application focused**: I'm not trying to build a library similar to langchain or llamaindex to perform data-driven operations on LLMs, I'm trying to build a very simple and lightweight framework that leverages the possibility of LLMs to perform function calling so that LLMs can easily be integrated in software applications.
- **Extensible**: I'm planning to add more providers and more features.

As a side note, I've just recently begun to employ Python for production code, therefore I might have "contaminated" the codebase with some approaches, patterns or choices that might not be idiomatic or "pythonic", I'm more than happy to receive feedback and open to suggestions on how to make the codebase cleaner and more idiomatic, hopefully without too many breaking changes.

## ⚠️ Project status

I'm using this library in many of my projects without problems, that said I've just revamped it entirely to support multiple providers and entirely refactored the codebase to streamline function calling. I've done my best to test it thoroughly, but I can't guarantee something won't break.

I'm actively working on this project and I'm open to contributions, so feel free to open an issue or a PR if you find something that needs fixing or improving.

My [next steps](docs/roadmap.md#-next---high-priority) will be to improve tests and documentation, and to extend the built in providers to support more models.

Since I've started the development of JAIms, a few similar projects have been started, and granted that I didn't have time to check them out yet, some might easily be more advanced, yet I've widely employed this library in my projects and those of the company I work for, and I've been actively maintaining it, so I'm planning to keep it up to date and to improve it as much as I can.

I've opted for an open source by default approach to allow others to benefit from it and force myself to keep the code clean and well documented, just remember that since this is, for now, a side-project developed just by me (that am fairly new to python), expect the possibility of encountering some issues and don't expect an immediate patch from me, any help is very much appreciated 🤗.


## 📝 License

Copyright (c) 2023 Marco Musella (aka Mush). This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
