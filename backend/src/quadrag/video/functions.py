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
        # Validate input
        if image is None:
            return "Invalid image: None provided"

        # Convert PIL Image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Validate base64 encoding
        if not img_base64:
            return "Failed to encode image"

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

        # Validate response
        if not response or not response.choices or len(response.choices) == 0:
            return "No response from vision API"

        description = response.choices[0].message.content
        if description and isinstance(description, str):
            description = description.strip()
            # Ensure we return a non-empty string
            return description if len(description) > 0 else "Empty description from API"
        else:
            return "Invalid response format from vision API"

    except Exception as e:
        # Return a safe fallback instead of raising an exception
        # This prevents the entire indexing process from failing
        error_msg = str(e)[:100]  # Truncate long error messages
        return f"Description unavailable: {error_msg}"

