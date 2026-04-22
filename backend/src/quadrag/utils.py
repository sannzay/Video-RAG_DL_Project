"""Utility functions for QuadRAG."""

import base64
import io
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import psutil
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
        
        # Transcode to H.264 Main profile with AAC audio - pixeltable compatible
        # -movflags +faststart for web streaming compatibility
        # -pix_fmt yuv420p for maximum compatibility
        # -b:a 128k for standard audio bitrate
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libx264",
                "-profile:v", "main",  # Force Main profile for pixeltable
                "-preset", "fast",
                "-pix_fmt", "yuv420p",  # Standard pixel format
                "-c:a", "aac",
                "-b:a", "128k",  # Standard audio bitrate
                "-strict", "experimental",
                "-movflags", "+faststart",
                "-y",  # Overwrite output
                output_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        # Validate the transcoded file
        if not Path(output_path).exists():
            raise RuntimeError(f"Transcoded file not created: {output_path}")

        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Transcoded file is empty: {output_path}")
        
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


# Large video support utilities

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or 0 if unable to determine
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)
        if "format" in data and "duration" in data["format"]:
            return float(data["format"]["duration"])
        elif "streams" in data:
            # Try to get duration from video stream
            for stream in data["streams"]:
                if stream.get("codec_type") == "video" and "duration" in stream:
                    return float(stream["duration"])

        logger.warning(f"Could not extract duration from ffprobe output for {video_path}")
        return 0.0

    except Exception as e:
        logger.warning(f"Failed to get video duration for {video_path}: {e}")
        return 0.0


def calculate_frame_count(video_duration_seconds: float) -> int:
    """Adaptive frame-sampling schedule.

    Calibrated around the two hard real-world constraints we've measured:

    * **Lazy domain-view build hits Railway's 60 s edge-proxy cap.**
      ``/chat`` with a fresh ``domain_context`` synchronously runs
      ``pxt_openai.vision`` (OpenRouter → gemini-2.0-flash) on every frame
      plus an embedding index, inside a single HTTP request. Empirically:
      ~0.5–0.7 s per vision call with Pixeltable's async concurrency
      + ~7 s for view creation/embedding index. 80 frames sits around
      ~55 s — the edge of the cliff. Above that, first-chat with a new
      lens starts hitting client timeouts, though the view still builds
      server-side and the second chat hits cache.

    * **Eager description-index build** (during upload's background thread)
      has a 30-minute budget, so it tolerates much higher frame counts
      without issue.

    The schedule below chooses the same N for both indexes — simpler, and
    keeps the lazy-build path predictable. If you push these much higher,
    also expect the first-query retry-on-cache-hit UX documented in the
    wizard spinner.
    """
    if video_duration_seconds <= 0:
        return 40  # Default fallback when ffprobe can't read duration

    if video_duration_seconds < 300:      # < 5 min
        return 40
    elif video_duration_seconds < 1800:   # < 30 min
        return 60
    elif video_duration_seconds < 3600:   # < 1 h
        return 80
    elif video_duration_seconds < 7200:   # < 2 h
        return 100
    else:                                  # very long videos
        # Roughly 1 frame per 90 s, floored at 100, capped at 150. Lazy
        # domain builds on videos this long may exceed the 60 s edge cap
        # on the first query — that's surfaced as "try again" in the UI.
        return min(150, max(100, int(video_duration_seconds // 90)))


def validate_video_size(file_path: str) -> bool:
    """Validate video file size and duration limits.

    Args:
        file_path: Path to video file

    Returns:
        True if valid

    Raises:
        ValueError: If video exceeds limits
    """
    MAX_FILE_SIZE_MB = 500  # Configurable limit
    MAX_DURATION_SECONDS = 7200  # 2 hours max

    # Check file size
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"Video too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit")

    # Check duration
    duration = get_video_duration(file_path)
    if duration > MAX_DURATION_SECONDS:
        hours = duration / 3600
        raise ValueError(f"Video too long: {hours:.1f} hours > {MAX_DURATION_SECONDS/3600:.1f} hour limit")

    logger.info(f"Video validation passed: {size_mb:.1f}MB, {duration:.1f}s duration")
    return True


@contextmanager
def monitor_processing(operation: str):
    """Context manager to monitor processing time and memory usage.

    Args:
        operation: Name of the operation being monitored
    """
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        logger.info(f"📊 {operation}: {duration:.1f}s, "
                   f"Memory: {start_memory:.1f}MB → {end_memory:.1f}MB "
                   f"(Δ{memory_delta:+.1f}MB)")


def cleanup_processing_files(original_path: str, transcoded_path: str) -> None:
    """Clean up intermediate processing files.

    Args:
        original_path: Path to original file
        transcoded_path: Path to transcoded file
    """
    try:
        transcoded_file = Path(transcoded_path)
        if transcoded_file.exists() and transcoded_file.stat().st_size > 0:
            # Transcoding succeeded, safe to remove original
            original_file = Path(original_path)
            if original_file.exists() and str(original_file) != str(transcoded_file):
                original_file.unlink()
                logger.info(f"Cleaned up original file: {original_path}")
        else:
            logger.warning(f"Transcoded file invalid or missing, keeping original: {original_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup files: {e}")
