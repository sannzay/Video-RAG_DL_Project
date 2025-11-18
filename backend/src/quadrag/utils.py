"""Utility functions for QuadRAG."""

import base64
import io
import json
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image
from loguru import logger


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize an image to the specified dimensions.

    Args:
        image: Input PIL Image
        width: Target width
        height: Target height

    Returns:
        Resized PIL Image
    """
    return image.resize((width, height), Image.Resampling.LANCZOS)


def extract_text_from_transcription(transcription_result: dict) -> str:
    """
    Extract text from transcription result.

    Args:
        transcription_result: Transcription dictionary from OpenAI

    Returns:
        Extracted text string
    """
    if isinstance(transcription_result, dict):
        return str(transcription_result.get("text", ""))
    return str(transcription_result)


def validate_video_format(video_path: str) -> bool:
    """
    Validate video format using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        True if video is in compatible format, False otherwise
    """
    try:
        logger.info(f"Validating video format for: {video_path}")
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile",
                "-of",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        info = json.loads(result.stdout or "{}")
        streams = info.get("streams", [])
        if not streams:
            logger.warning("No video streams detected during validation")
            return True

        stream = streams[0]
        codec = (stream.get("codec_name") or "").lower()
        profile = (stream.get("profile") or "").lower()

        logger.info(f"Video codec={codec or 'unknown'}, profile={profile or 'unknown'}")

        if codec == "h264" and profile.startswith("high"):
            logger.warning("Video uses H.264 High profile (incompatible). Needs transcoding.")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating video format: {e}")
        return False


def transcode_video(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Transcode video to a more compatible format (H.264 Main profile).

    Args:
        input_path: Path to input video
        output_path: Path for output video (default: input_path + "_transcoded.mp4")

    Returns:
        Path to transcoded video

    Raises:
        RuntimeError: If transcoding fails
    """
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_transcoded.mp4")
    
    try:
        logger.info(f"Transcoding video from {input_path} to {output_path}")
        
        # Transcode to H.264 Main profile with AAC audio
        # -movflags +faststart for web streaming compatibility
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libx264",
                "-profile:v", "main",  # Force Main profile
                "-preset", "fast",
                "-c:a", "aac",
                "-strict", "experimental",
                "-movflags", "+faststart",
                "-y",  # Overwrite output
                output_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Successfully transcoded video to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg transcoding failed: {e.stderr}")
        raise RuntimeError(f"Video transcoding failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error transcoding video: {e}")
        raise RuntimeError(f"Video transcoding failed: {str(e)}")


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded image string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image object
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))
