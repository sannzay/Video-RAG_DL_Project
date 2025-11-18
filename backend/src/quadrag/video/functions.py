"""Pixeltable UDF functions for video processing."""

import pixeltable as pxt
from PIL import Image


@pxt.udf
def resize_image(image: pxt.type_system.Image, width: int, height: int) -> pxt.type_system.Image:
    """
    Resize an image to fit within the specified width and height while maintaining aspect ratio.
    Note: The PIL.Image.thumbnail() method modifies the image in place.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")

    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return img_copy


@pxt.udf
def extract_text_from_chunk(transcript: pxt.type_system.Json) -> str:
    """
    Extract text from a transcript JSON object.
    Note: Predictions of common S2T models are in dict format containing the text and chunk timestamps metadata. We need the text only.
    """
    if isinstance(transcript, dict):
        return str(transcript.get("text", ""))
    return str(transcript)

