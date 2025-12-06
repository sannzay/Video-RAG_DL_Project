"""Pixeltable UDF functions for video processing."""

import base64
from io import BytesIO
import pixeltable as pxt
from PIL import Image
import openai


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


@pxt.udf
def describe_image(image: pxt.type_system.Image) -> str:
    """
    Generate a detailed description of an image using OpenAI's Vision API.

    This is a synchronous UDF that bypasses Pixeltable's async context to avoid
    event loop conflicts with FastAPI/uvloop.

    Args:
        image: PIL Image to describe

    Returns:
        str: Detailed description of the image content
    """
    try:
        # Convert PIL Image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Create OpenAI client (will use environment variable OPENAI_API_KEY)
        client = openai.OpenAI()

        # Make synchronous API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what is happening in this image in detail. Be specific about objects, people, actions, and setting."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }],
            max_tokens=200,
            temperature=0.3
        )

        description = response.choices[0].message.content.strip()
        return description if description else "Unable to generate description"

    except Exception as e:
        # Return a safe fallback instead of raising an exception
        # This prevents the entire indexing process from failing
        return f"Description unavailable: {str(e)}"

