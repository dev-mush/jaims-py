from PIL import Image
import base64
import io

mime_type_mapping = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def image_to_bytes(image: Image.Image) -> tuple[str, bytes]:

    if image.format not in mime_type_mapping:
        raise ValueError(f"Unsupported image format: {image.format}")

    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return mime_type_mapping[image.format], buffered.getvalue()


def image_to_b64(image: Image.Image) -> tuple[str, str]:

    img_bytes = image_to_bytes(image)
    return img_bytes[0], base64.b64encode(img_bytes[1]).decode("utf-8")
