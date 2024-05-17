import os
from jaims import (
    JAImsImageContent,
    JAImsMessage,
)

from jaims.adapters.openai_adapter import (
    JAImsOpenaiKWArgs,
    create_jaims_openai,
    JAImsGPTModel,
)

from jaims.adapters.google_generative_ai_adapter import (
    create_jaims_gemini,
    GEMINI_1_PRO_VISION_LATEST,
)

from PIL import Image

from jaims.entities import JAImsMessageRole


def main():
    stream = False

    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "image.png")
    pil_image = Image.open(image_path)
    image_url = "https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9"

    openai_agent = create_jaims_openai(
        kwargs=JAImsOpenaiKWArgs(
            model=JAImsGPTModel.GPT_4_VISION_PREVIEW,
            stream=stream,
        ),
    )

    gemini_agent = create_jaims_gemini(
        model=GEMINI_1_PRO_VISION_LATEST,
    )

    messages = [
        JAImsMessage(
            role=JAImsMessageRole.USER,
            contents=[
                "Are these images the same? What do they contain?",
                JAImsImageContent(pil_image),
                JAImsImageContent(image_url),
            ],
        )
    ]

    # openai_response = openai_agent.run(
    #     [
    #         JAImsMessage(
    #             role=JAImsMessageRole.USER,
    #             contents=[
    #                 "Are these images the same? What do they contain?",
    #                 JAImsImageContent(pil_image),
    #                 JAImsImageContent(image_url),
    #             ],
    #         )
    #     ]
    # )

    # # print("OpenAI Response:")
    # # print(openai_response)

    gemini_response = gemini_agent.run(
        [
            JAImsMessage(
                role=JAImsMessageRole.USER,
                contents=[
                    "What does this image contain?",
                    JAImsImageContent(pil_image),
                ],
            )
        ]
    )

    print("Gemini Response:")
    print(gemini_response)


if __name__ == "__main__":
    main()
