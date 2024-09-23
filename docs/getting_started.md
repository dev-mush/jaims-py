# JAIms - Getting Started Guide

## Installation

To install JAIms, use pip:

```bash
pip install jaims-py
```

Note: Installing `jaims-py` alone doesn't include the built-in providers (this is a deliberate choice to avoid overcluttering dependencies in a specific python project). To use specific providers, you need to install their dependencies separately.

For example, to install OpenAI support:

```bash
pip install jaims-py[openai]
```

You can install multiple providers at once:

```bash
pip install jaims-py[openai,google,mistral,anthropic]
```

## Basic Usage

Here's a simple example using OpenAI's GPT-4 model:

```python
from jaims import Agent, Message, LLMParams, Config

# Create an agent
agent = Agent.build(
    model="gpt-4o",
    provider="openai",
    llm_params=LLMParams(temperature=0.7, max_tokens=150)
)

# Send a message and get a response
user_message = Message.user_message("Hello, can you tell me a joke?")
response = agent.message([user_message])

print(response)
```

## Function Calling Example

JAIms supports function calling. Here's a simple example:

```python
from jaims import Agent, Message, jaimsfunctiontool

# Define a function tool
@jaimsfunctiontool(
    description="Calculates the sum of two numbers",
    params_descriptions={"a": "first number", "b": "second number"}
)
def add_numbers(a: int, b: int):
    return a + b

# Create an agent with the function tool
agent = Agent.build(
    model="gpt-4",
    provider="openai",
    tools=[add_numbers]
)

# Use the function in a conversation
user_message = Message.user_message("What's the sum of 5 and 7?")
response = agent.message([user_message])

print(response)
```

In this example, the agent can use the `add_numbers` function when appropriate to perform calculations.

## Fast Calling Example

JAIms also supports fast calling for quick, one-off function executions. Here's an example:

```python
from jaims import Agent, Message, BaseModel, Field, FunctionToolDescriptor

# Define a model for the function output
class SumResult(BaseModel):
    result: int = Field(description="The sum of the two numbers")

# Create a function tool descriptor
sum_numbers = FunctionToolDescriptor(
    name="sum_numbers",
    description="Calculates the sum of two numbers",
    params=SumResult
)

# Use fast calling to execute the function
result = Agent.run_tool_model(
    model="gpt-4",
    provider="openai",
    descriptor=sum_numbers,
    messages=[Message.user_message("What's the sum of 10 and 15?")]
)

print(f"The sum is: {result.result}")
```

This example demonstrates how to use fast calling to quickly execute a function and get a structured result.

## Next Steps

- Explore more advanced features like streaming responses and complex function tools
- Check out the full documentation for detailed information on all features