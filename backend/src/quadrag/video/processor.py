"""Video processor for creating and managing video indexes."""

import uuid
from pathlib import Path
from typing import Optional

import pixeltable as pxt
from loguru import logger

from quadrag.config import get_settings
from quadrag.video.registry import (
    add_video_to_registry,
    get_video_from_registry,
    video_exists_in_registry,
)

settings = get_settings()
logger = logger.bind(name="VideoProcessor")


class VideoProcessor:
    """Processes videos and creates Pixeltable indexes."""

    def __init__(self):
        """Initialize the video processor."""
        self.settings = settings
        logger.info(
            f"VideoProcessor initialized\n"
            f"  Frame count: {settings.SPLIT_FRAMES_COUNT}\n"
            f"  Audio chunk: {settings.AUDIO_CHUNK_LENGTH}s"
        )

    def process_video(self, video_id: str, video_path: str) -> bool:
        """Process a video and create base indexes (Image, Audio, Description).

        Args:
            video_id: Unique video identifier
            video_path: Path to the video file

        Returns:
            True if processing was successful

        Raises:
            ValueError: If video processing fails
        """
        try:
            # Check if video already exists
            if video_exists_in_registry(video_id):
                logger.info(f"Video {video_id} already exists in registry")
                return True

            # Create cache directory for this video
            cache_dir = f"video_{uuid.uuid4().hex[:8]}"
            cache_path = settings.get_cache_dir() / cache_dir
            cache_path.mkdir(parents=True, exist_ok=True)

            # Create Pixeltable directory
            pxt_dir = f"{cache_dir}"
            pxt.create_dir(pxt_dir, if_exists="replace_force")

            # Create table names
            video_table_name = f"{pxt_dir}.video_table"
            frames_view_name = f"{video_table_name}_frames"
            audio_view_name = f"{video_table_name}_audio"
            description_view_name = f"{video_table_name}_descriptions"

            # Create video table
            logger.info(f"Creating video table: {video_table_name}")
            video_table = pxt.create_table(
                video_table_name,
                schema={"video": pxt.Video},
                if_exists="replace_force",
            )

            # Insert video with absolute path
            logger.info(f"Inserting video: {video_path}")
            abs_video_path = str(Path(video_path).absolute())
            logger.info(f"Using absolute path: {abs_video_path}")
            video_table.insert([{"video": abs_video_path}])

            # Add to registry
            add_video_to_registry(
                video_id=video_id,
                cache_dir=cache_dir,
                video_table_name=video_table_name,
                frames_view_name=frames_view_name,
                audio_view_name=audio_view_name,
                description_view_name=description_view_name,
            )

            logger.info(f"Successfully processed video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            raise ValueError(f"Video processing failed: {e}")

    def get_video_info(self, video_id: str):
        """Get video index information.

        Args:
            video_id: Video identifier

        Returns:
            VideoIndexInfo object or None
        """
        return get_video_from_registry(video_id)

    def video_exists(self, video_id: str) -> bool:
        """Check if video has been processed.

        Args:
            video_id: Video identifier

        Returns:
            True if video exists
        """
        return video_exists_in_registry(video_id)


