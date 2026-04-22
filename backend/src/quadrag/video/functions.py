"""Pixeltable UDF functions for video processing.

After Step 9 the frame-description and domain-caption UDFs were retired in
favor of Pixeltable's native ``pxt_openai.vision``. Post-0.3.0 those were
brought back (as ``describe_via_openrouter``) so vision calls can flow
through OpenRouter — OpenAI's Tier-1 TPM was too tight for back-to-back
description + domain indexing, and OpenRouter's provider rate-limits are
more forgiving. ``pxt_openai.transcriptions`` + ``pxt_openai.embeddings``
still go direct to OpenAI (OpenRouter doesn't expose those endpoints).

UDFs here:

* ``resize_image`` — used while building the frames view.
* ``extract_text_from_chunk`` — pulls the ``text`` field out of Whisper's
  JSON response. The underlying ``_extract_text_from_chunk`` is plain Python
  so unit tests can exercise it without going through Pixeltable's column
  expression machinery.
* ``describe_via_openrouter`` — async UDF that captions an image through
  OpenRouter. Takes the image and a prompt; model is passed via kwarg so
  callers can pick ``google/gemini-2.0-flash-001`` (cheap), Claude, etc.
"""

import base64
from io import BytesIO

import openai
import pixeltable as pxt
from PIL import Image

from quadrag.config import get_settings

_settings = get_settings()


def _openrouter_client() -> openai.AsyncOpenAI:
    """Lazy, singleton-per-worker async OpenAI client pointed at OpenRouter.

    Kept as a module-level factory so Pixeltable's async UDF runner can share
    the HTTP connection pool across concurrent vision calls.
    """
    return openai.AsyncOpenAI(
        base_url=_settings.OPENROUTER_BASE_URL,
        api_key=_settings.OPENROUTER_API_KEY,
    )


@pxt.udf
def resize_image(image: pxt.type_system.Image, width: int, height: int) -> pxt.type_system.Image:
    """Resize an image to fit within ``width``×``height`` while preserving aspect ratio."""
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")

    # Create a copy to avoid modifying the original.
    # PIL.Image.thumbnail() modifies in place, hence the copy.
    img_copy = image.copy()
    img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return img_copy


def _extract_text_from_chunk(transcript) -> str:
    """Pure-Python impl of Whisper JSON → text extraction.

    Separated from the Pixeltable UDF so unit tests can exercise it directly
    without routing through Pixeltable's column-expression machinery.
    """
    if isinstance(transcript, dict):
        return str(transcript.get("text", ""))
    if transcript is None:
        return ""
    return str(transcript)


@pxt.udf
def extract_text_from_chunk(transcript: pxt.type_system.Json) -> str:
    """Extract the ``text`` field from a Whisper transcript JSON object.

    Whisper predictions come back as a dict containing the full transcript
    plus chunk-level timestamp metadata; we only want the transcript text
    for embedding and display.
    """
    return _extract_text_from_chunk(transcript)


@pxt.udf
async def describe_via_openrouter(
    image: pxt.type_system.Image,
    prompt: str,
    model: str,
    max_tokens: int = 200,
    temperature: float = 0.3,
) -> str:
    """Caption an image via OpenRouter's OpenAI-compatible chat/completions API.

    Async so Pixeltable's runtime can run many calls concurrently. We rely on
    OpenRouter's provider-level rate limiting — if gpt-4o-mini is saturated,
    switch to ``google/gemini-2.0-flash-001`` or Claude Haiku by changing
    ``settings.IMAGE_CAPTION_MODEL``.

    Returns a safe fallback string on API failure rather than raising, so a
    single bad frame doesn't abort the whole indexing pass.
    """
    try:
        if image is None:
            return "Invalid image: None provided"

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        if not img_b64:
            return "Failed to encode image"

        client = _openrouter_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not response or not response.choices or len(response.choices) == 0:
            return "No response from OpenRouter vision API"
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            content = content.strip()
            return content if content else "Empty response from OpenRouter"
        return "Invalid response format from OpenRouter"

    except Exception as e:
        # Truncate so an OpenRouter 429 message doesn't balloon per-frame.
        error_msg = str(e)[:150]
        return f"Vision unavailable: {error_msg}"
