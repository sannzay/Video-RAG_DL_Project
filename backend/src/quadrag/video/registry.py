"""Video index registry for tracking processed videos."""

import json
from pathlib import Path
from typing import Dict, Optional

import pixeltable as pxt
from loguru import logger

# Global registry to store video indexes
_VIDEO_REGISTRY: Dict[str, "VideoIndexInfo"] = {}
_REGISTRY_FILE = Path("../data/cache/video_registry.json")


class VideoIndexInfo:
    """Information about a video's Pixeltable indexes."""

    def __init__(
        self,
        video_id: str,
        cache_dir: str,
        video_table_name: str,
        frames_view_name: str,
        audio_view_name: str,
        description_view_name: str,
        domain_view_name: Optional[str] = None,
        domain_captions: Optional[Dict[float, str]] = None,
        domain_context: Optional[str] = None,
    ):
        self.video_id = video_id
        self.cache_dir = cache_dir
        self.video_table_name = video_table_name
        self.frames_view_name = frames_view_name
        self.audio_view_name = audio_view_name
        self.description_view_name = description_view_name
        self.domain_view_name = domain_view_name
        self.domain_captions = domain_captions or {}
        self.domain_context = domain_context

    @property
    def video_table(self):
        """Get the video table."""
        return pxt.get_table(self.video_table_name)

    @property
    def frames_view(self):
        """Get the frames view."""
        return pxt.get_table(self.frames_view_name)

    @property
    def audio_view(self):
        """Get the audio chunks view."""
        return pxt.get_table(self.audio_view_name)

    @property
    def description_view(self):
        """Get the description view."""
        return pxt.get_table(self.description_view_name)

    @property
    def domain_view(self):
        """Get the domain captions view."""
        if self.domain_view_name:
            return pxt.get_table(self.domain_view_name)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "video_id": self.video_id,
            "cache_dir": self.cache_dir,
            "video_table_name": self.video_table_name,
            "frames_view_name": self.frames_view_name,
            "audio_view_name": self.audio_view_name,
            "description_view_name": self.description_view_name,
            "domain_view_name": self.domain_view_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoIndexInfo":
        """Create from dictionary."""
        return cls(**data)


def add_video_to_registry(
    video_id: str,
    cache_dir: str,
    video_table_name: str,
    frames_view_name: str,
    audio_view_name: str,
    description_view_name: str,
    domain_view_name: Optional[str] = None,
) -> VideoIndexInfo:
    """Add a video to the registry.

    Args:
        video_id: Unique video identifier
        cache_dir: Pixeltable cache directory
        video_table_name: Name of the video table
        frames_view_name: Name of the frames view
        audio_view_name: Name of the audio view
        description_view_name: Name of the description view
        domain_view_name: Optional name of the domain captions view

    Returns:
        VideoIndexInfo object
    """
    info = VideoIndexInfo(
        video_id=video_id,
        cache_dir=cache_dir,
        video_table_name=video_table_name,
        frames_view_name=frames_view_name,
        audio_view_name=audio_view_name,
        description_view_name=description_view_name,
        domain_view_name=domain_view_name,
    )
    _VIDEO_REGISTRY[video_id] = info
    _save_registry()
    logger.info(f"Added video {video_id} to registry")
    return info


def get_video_from_registry(video_id: str) -> Optional[VideoIndexInfo]:
    """Get video info from registry.

    Args:
        video_id: Video identifier

    Returns:
        VideoIndexInfo or None if not found
    """
    return _VIDEO_REGISTRY.get(video_id)


def video_exists_in_registry(video_id: str) -> bool:
    """Check if video exists in registry.

    Args:
        video_id: Video identifier

    Returns:
        True if video exists in registry
    """
    return video_id in _VIDEO_REGISTRY


def update_domain_view(video_id: str, domain_view_name: str) -> None:
    """Update the domain view name for a video.

    Args:
        video_id: Video identifier
        domain_view_name: Name of the domain captions view
    """
    if video_id in _VIDEO_REGISTRY:
        _VIDEO_REGISTRY[video_id].domain_view_name = domain_view_name
        _save_registry()
        logger.info(f"Updated domain view for video {video_id}")


def update_domain_captions(video_id: str, domain_captions: Dict[float, str], domain_context: str) -> None:
    """Update the domain captions for a video.

    Args:
        video_id: Video identifier
        domain_captions: Dictionary mapping pos_msec to captions
        domain_context: The domain context used
    """
    if video_id in _VIDEO_REGISTRY:
        _VIDEO_REGISTRY[video_id].domain_captions = domain_captions
        _VIDEO_REGISTRY[video_id].domain_context = domain_context
        _save_registry()
        logger.info(f"Updated domain captions for video {video_id} ({len(domain_captions)} captions)")


def get_all_videos() -> Dict[str, VideoIndexInfo]:
    """Get all videos from registry.

    Returns:
        Dictionary of video_id to VideoIndexInfo
    """
    return _VIDEO_REGISTRY.copy()


def _save_registry() -> None:
    """Save registry to disk."""
    try:
        _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_REGISTRY_FILE, "w") as f:
            data = {vid: info.to_dict() for vid, info in _VIDEO_REGISTRY.items()}
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")


def _load_registry() -> None:
    """Load registry from disk."""
    global _VIDEO_REGISTRY
    try:
        if _REGISTRY_FILE.exists():
            with open(_REGISTRY_FILE, "r") as f:
                data = json.load(f)
                _VIDEO_REGISTRY = {
                    vid: VideoIndexInfo.from_dict(info_dict)
                    for vid, info_dict in data.items()
                }
            logger.info(f"Loaded {len(_VIDEO_REGISTRY)} videos from registry")
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        _VIDEO_REGISTRY = {}


# Load registry on module import
_load_registry()


