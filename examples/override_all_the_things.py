import json
from typing import Any, Dict, List
from jaims import (
    JAImsAgent,
    JAImsFunctionTool,
    JAImsToolCall,
    JAImsMessage,
    JAImsParamDescriptor,
    JAImsFunctionToolDescriptor,
    JAImsJsonSchemaType,
    JAImsToolManager,
    JAImsDefaultHistoryManager,
)

from jaims.adapters.openai_adapter import (
    create_jaims_openai,
    JAImsOpenaiKWArgs,
    JAImsOptions,
    JAImsOpenaiAdapter,
)

from examples_utils import FileTransactionStorage


"""
This example shows how to customize classes and how to override parameters
in between agent runs.
"""


class MyCustomToolManager(JAImsToolManager):

    def handle_tool_calls(
        self,
        agent: JAImsAgent,
        tool_calls: List[JAImsToolCall],
        tools: List[JAImsFunctionTool],
    ) -> List[JAImsMessage]:
        response_messages = []
        for tool_call in tool_calls:

            if tool_call.tool_name == "echo":
                if tool_call.tool_args:
                    response_messages.append(
                        JAImsMessage.tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            response=tool_call.tool_args.get("value", ""),
                        )
                    )
                else:
                    response_messages.append(
                        JAImsMessage.tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            response="error: Value param not passed",
                        )
                    )
            elif tool_call.tool_name == "reverse":
                if tool_call.tool_args:
                    value = tool_call.tool_args.get("value", "")
                    result = value[::-1]
                    response_messages.append(
                        JAImsMessage.tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            response=result,
                        )
                    )
                else:
                    response_messages.append(
                        JAImsMessage.tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            response="error: Value param not passed",
                        )
                    )
            elif tool_call.tool_name == "change_model":
                openai_adapter = agent.llm_interface
                assert isinstance(openai_adapter, JAImsOpenaiAdapter)
                current_kwargs = openai_adapter.kwargs
                if isinstance(current_kwargs, dict):
                    current_kwargs = JAImsOpenaiKWArgs.from_dict(current_kwargs)

                new_model = (
                    "gpt-3.5-turbo"
                    if current_kwargs.model == "gpt-4-turbo"
                    else "gpt-4-turbo"
                )

                openai_adapter.kwargs = current_kwargs.copy_with_overrides(
                    model=new_model
                )
                agent.llm_interface = openai_adapter
                response_messages.append(
                    JAImsMessage.tool_response_message(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        response=f"Model changed to {new_model}",
                    )
                )

        return response_messages


def main():

    echo_wrapper = JAImsFunctionTool(
        JAImsFunctionToolDescriptor(
            name="echo",
            description="use this function when the user asks you to echo some string",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="value",
                    description="the string to echo",
                    json_type=JAImsJsonSchemaType.STRING,
                )
            ],
        ),
    )

    reverse_wrapper = JAImsFunctionTool(
        JAImsFunctionToolDescriptor(
            name="reverse",
            description="use this function when the user asks you to reverse some string",
            params_descriptors=[
                JAImsParamDescriptor(
                    name="value",
                    description="the string to reverse",
                    json_type=JAImsJsonSchemaType.STRING,
                )
            ],
        ),
    )

    change_model_wrapper = JAImsFunctionTool(
        JAImsFunctionToolDescriptor(
            name="change_model",
            description="use this function when the user asks you to change the current model",
            params_descriptors=[],
        ),
    )

    agent = create_jaims_openai(
        kwargs=JAImsOpenaiKWArgs(
            model="gpt-4-turbo",
        ),
        history_manager=JAImsDefaultHistoryManager(
            leading_prompts=[
                JAImsMessage.system_message(
                    "You are JAIms, a helpful assistant that calls the tools that the user asks to call."
                )
            ]
        ),
        tools=[
            echo_wrapper,
            reverse_wrapper,
            change_model_wrapper,
        ],
        transaction_storage=FileTransactionStorage(),
        tool_manager=MyCustomToolManager(),
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.run_stream(
            [JAImsMessage.user_message(user_input)],
        )

        for chunk in response:
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
