import json
from typing import Any, Dict, List
from jaims import (
    JAImsAgent,
    JAImsFunctionTool,
    JAImsParamDescriptor,
    JAImsFunctionToolDescriptor,
    JAImsJsonSchemaType,
    JAImsGPTModel,
    JAImsOpenaiKWArgs,
    JAImsOptions,
    JAImsDefaultToolManager,
    JAImsToolResults,
)


"""
This example shows how to customize classes and how to override parameters
in between agent runs.
"""


class MyCustomToolHandler(JAImsDefaultToolManager):

    def handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        function_wrappers: List[JAImsFunctionTool],
        current_kwargs: JAImsOpenaiKWArgs,
        current_options: JAImsOptions,
    ) -> JAImsToolResults:

        response_messages = []
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            result = "tool correctly executed"
            override_kwargs = None
            stop = False

            if function_name == "echo":
                value = function_args.get("value", "")
                result = f"echoing {value}"
            elif function_name == "reverse":
                value = function_args.get("value", "")
                result = value[::-1]
            elif function_name == "change_model":
                new_model = (
                    JAImsGPTModel.GPT_3_5_TURBO_0613
                    if current_kwargs.model == JAImsGPTModel.GPT_4_0613
                    else JAImsGPTModel.GPT_4_0613
                )
                override_kwargs = current_kwargs.copy_with_overrides(model=new_model)
                result = f"model changed to {new_model.value}"
            elif function_name == "stop_operations":
                stop = False

            response_messages.append(
                {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": result,
                }
            )

        return JAImsToolResults(
            function_result_messages=response_messages,
            stop=stop,
            override_kwargs=override_kwargs,
        )


def main():
    stream = False

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

    stop_operations_wrapper = JAImsFunctionTool(
        JAImsFunctionToolDescriptor(
            name="stop_operations",
            description="use this function when the user asks you to stop all the operations",
            params_descriptors=[],
        ),
    )

    agent = JAImsAgent(
        options=JAImsOptions(
            leading_prompts=[
                {
                    "role": "system",
                    "content": "You are JAIms, a helpful assistant that calls the tools that the user asks to call.",
                }
            ],
        ),
        openai_kwargs=JAImsOpenaiKWArgs(
            model=JAImsGPTModel.GPT_3_5_TURBO_0613,
            stream=stream,
            tools=[
                echo_wrapper,
                reverse_wrapper,
                change_model_wrapper,
                stop_operations_wrapper,
            ],
        ),
        tool_manager=MyCustomToolHandler(),
    )

    print("Hello, I am JAIms, your personal assistant.")
    print("How can I help you today?")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        response = agent.run(
            [{"role": "user", "content": user_input}],
        )

        if response:
            if stream:
                for chunk in response:
                    print(chunk, end="", flush=True)
                print("\n")

            else:
                print(response)

    expenses = agent.get_expenses()
    for expense in expenses:
        print(expense)


if __name__ == "__main__":
    main()
