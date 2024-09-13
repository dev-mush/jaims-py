from typing import List
from jaims import (
    Agent,
    FunctionTool,
    ToolCall,
    Message,
    FunctionToolDescriptor,
    ToolManagerITF,
    DefaultHistoryManager,
    ToolResponse,
    create_model,
    Field,
)

from jaims.adapters.openai_adapter import (
    OpenAIParams,
    OpenaiAdapter,
)


class MyCustomToolManager(ToolManagerITF):

    def __init__(self) -> None:
        self.agent = None

    def bind_agent(self, agent: Agent) -> None:
        self.agent = agent

    def handle_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tools: List[FunctionTool],
    ) -> List[ToolResponse]:
        tool_responses = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "echo":
                print("echo called")
                if tool_call.tool_args:
                    response_str = tool_call.tool_args.get("value", "")
                    is_error = False
                else:
                    response_str = "error: Value param not passed"
                    is_error = True

                tool_responses.append(
                    ToolResponse(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        response=response_str,
                        is_error=is_error,
                    )
                )
            elif tool_call.tool_name == "reverse":
                print("reverse called")
                if tool_call.tool_args:
                    value = tool_call.tool_args.get("value", "")
                    response_str = value[::-1]
                    is_error = False
                else:
                    response_str = "error: Value param not passed"
                    is_error = True

                tool_responses.append(
                    ToolResponse(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        response=response_str,
                        is_error=is_error,
                    )
                )
            elif tool_call.tool_name == "change_model":
                if self.agent is None:
                    raise ValueError("Agent not bound to the tool manager")

                openai_adapter = self.agent.llm_interface
                assert isinstance(openai_adapter, OpenaiAdapter)
                current_kwargs = openai_adapter.kwargs
                if isinstance(current_kwargs, dict):
                    current_kwargs = OpenAIParams.from_dict(current_kwargs)

                new_model = (
                    "gpt-4o" if current_kwargs.model == "gpt-4-turbo" else "gpt-4-turbo"
                )

                openai_adapter.kwargs = current_kwargs.copy_with_overrides(
                    model=new_model
                )
                self.agent.llm_interface = openai_adapter
                tool_responses.append(
                    ToolResponse(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        response=f"Model changed to {new_model}",
                    )
                )

                print("Model changed to", new_model)

        return tool_responses


def main():

    echo_wrapper = FunctionTool(
        descriptor=FunctionToolDescriptor(
            name="echo",
            description="use this function when the user asks you to echo some string",
            params=create_model(
                "echo_params", value=(str, Field(description="the string to echo"))
            ),
        ),
    )

    reverse_wrapper = FunctionTool(
        FunctionToolDescriptor(
            name="reverse",
            description="use this function when the user asks you to reverse some string",
            params=create_model(
                "reverse_params",
                value=(str, Field(description="the string to reverse")),
            ),
        ),
    )

    change_model_wrapper = FunctionTool(
        FunctionToolDescriptor(
            name="change_model",
            description="use this function when the user asks you to change the current model",
            params=None,
        ),
    )

    tool_manager = MyCustomToolManager()

    agent = Agent.build(
        model="gpt-4-turbo",
        provider="openai",
        tool_manager=tool_manager,
        history_manager=DefaultHistoryManager(
            leading_prompts=[
                Message.system_message(
                    "You are JAIms, a helpful assistant that calls the tools that the user asks to call."
                )
            ]
        ),
        tools=[
            echo_wrapper,
            reverse_wrapper,
            change_model_wrapper,
        ],
    )

    tool_manager.bind_agent(agent)

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.message_stream(
            [Message.user_message(user_input)],
        )

        for chunk in response:
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
