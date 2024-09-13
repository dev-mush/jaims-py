import os
from jaims import ImageContent, Message

from jaims.adapters.openai_adapter import (
    JAImsOpenaiKWArgs,
    create_jaims_openai,
)

from jaims.adapters.google_generative_ai_adapter import (
    create_jaims_gemini,
)

from PIL import Image

from jaims.entities import MessageRole


def main():
    stream = False

    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "image.png")
    pil_image = Image.open(image_path)
    image_url = "https://github.com/dev-mush/jaims-py/assets/669003/5c53381f-25b5-4141-bcd2-7457863eafb9"

    openai_agent = create_jaims_openai(
        kwargs=JAImsOpenaiKWArgs(
            model="gpt-4-turbo",
            stream=stream,
        ),
    )

    gemini_agent = create_jaims_gemini(
        model="gemini-1.5-flash",
    )

    openai_response = openai_agent.message(
        [
            Message(
                role=MessageRole.USER,
                contents=[
                    "Are these images the same? What do they contain?",
                    ImageContent(pil_image),
                    ImageContent(image_url),
                ],
            )
        ]
    )

    print("OpenAI Response:")
    print(openai_response)

    gemini_response = gemini_agent.message(
        [
            Message(
                role=MessageRole.USER,
                contents=[
                    "What does this image contain?",
                    ImageContent(pil_image),
                ],
            )
        ]
    )

    print("Gemini Response:")
    print(gemini_response)


if __name__ == "__main__":
    main()
