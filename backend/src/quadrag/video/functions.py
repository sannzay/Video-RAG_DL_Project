"""Pixeltable UDF functions for video processing.

After Step 9 the frame-description and domain-caption UDFs were retired in
favor of Pixeltable's native ``pxt_openai.vision``, which runs API calls
asynchronously with adaptive rate-limit throttling. The two UDFs that remain
here are pure-Python helpers that don't touch external services:

* ``resize_image`` — used while building the frames view.
* ``extract_text_from_chunk`` — pulls the ``text`` field out of Whisper's
  JSON response. The underlying ``_extract_text_from_chunk`` is plain Python
  so unit tests can exercise it without going through Pixeltable's column
  expression machinery.
"""

import pixeltable as pxt
from PIL import Image


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
