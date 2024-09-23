<p align="center">
    <img width="300" src="https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9" >
</p>

# JAIms

_My name is Bot, JAIms Bot._ 🕶️

JAIms is a lightweight Python package that lets you build powerful LLM-Based agents or LLM powered applications with ease. It is platform agnostic, so you can focus on integrating AI into your software and let JAIms handle the boilerplate of communicating with the LLM API.
The main goal of JAIms is to provide a simple and easy-to-use interface to leverage the power of LLMs in your software, without having to worry about the specifics of the underlying provider, and to seamlessly integrate LLM functionality with your own codebase.
JAIms currently supports mainstream foundation LLMs such as OpenAI's GPT models, Google's gemini models (also on Vertex), Mistral models and Anthropic Models (both hosted on Anthropic and Vertex endpoints). JAIms can be easily extended to connect to your own model and endpoints.

Check out the [getting started guide](docs/getting_started.md) to quickly get up and running with JAIms.

Also consider checking out the [examples](examples) folder for more advanced use cases.

### ✨ Main Features

- Built in support for most common foundational models.
- Built in conversation history management to allow fast creation of chatbots, this can be easily extended to support more advanced history management strategies.
- Image support for multimodal LLMs 🖼️.
- Support for function calling, both streamed and non-streamed.
- Fast integration with dataclasses and pydantic models.
- Error handling and exponential backoff for built in providers (openai, google, mistral)

### 🧠 Guiding Principles

JAIms comes out of the necessity for a lightweight and easy-to-use framework to create LLM agents or integrate LLM functionality in python projects. Given the increasing work with both foundational and open source LLMs, JAIms has been designed as an abstraction layer to streamline fast creation of agentic business logic and seamless codebase integration.

In case you like to contribute, please keep in mind that I try to keep the code:

- **Modular**: any component is provided with a default basic implementation and an interface that can be easily extended for more complex use cases.
- **Type Hinted and Explicit**: I've done my best to type hint everything and document the codebase as much as possible to avoid digging into the code.
- **Tested**: Well...Let's just say I could have done better, but am planning to improve code coverage and test automation in the near future.
- **Application focused**: I'm not trying to build a library similar to langchain or llamaindex to perform data-driven operations on LLMs, I'm trying to build a very simple and lightweight framework that leverages the possibility of LLMs to perform function calling so that LLMs can easily be integrated in software applications.
- **Extensible**: I'm planning to add more providers and more features.

## ⚠️ Project status

I'm using this library in many of my projects without problems, that said I've just revamped it entirely to support multiple providers and entirely refactored the codebase to streamline function calling. I've done my best to test it thoroughly, but I can't guarantee something won't break.

In the [roadmap](docs/roadmap.md) I'm tracking the next steps I'm planning to take to improve the library.

I'm actively working on this project and I'm open to contributions, so feel free to open an issue or a PR if you find something that needs fixing or improving.

Since I've started the development of JAIms, a few similar projects have been started, and granted that I didn't have time to check them out yet, some might easily be more advanced, yet I've widely employed this library in my projects and those of the company I work for, and I've been actively maintaining it, so I'm planning to keep it up to date and to improve it as much as I can.

I've opted for an open source by default approach to allow others to benefit from it and force myself to keep the code clean and well documented, just remember that since this is, for now, a side-project developed just by me (that am fairly new to python), expect the possibility of encountering some issues and don't expect an immediate patch from me, any help is very much appreciated 🤗.

## 📝 License

Copyright (c) 2023 Marco Musella (aka Mush). This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
