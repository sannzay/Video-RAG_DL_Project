"""Configuration settings for QuadRAG."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
    )

    # === API Keys ===
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str

    # === Model Configuration ===
    AUDIO_TRANSCRIPT_MODEL: str = "whisper-1"
    IMAGE_CAPTION_MODEL: str = "gpt-4o-mini"
    TEXT_EMBEDDING_MODEL: str = "models/text-embedding-004"
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # === Video Processing Configuration ===
    SPLIT_FRAMES_COUNT: int = 45
    AUDIO_CHUNK_LENGTH: int = 10  # Seconds per chunk
    AUDIO_OVERLAP_SECONDS: int = 1
    AUDIO_MIN_CHUNK_DURATION_SECONDS: int = 1
    # Note: For a 5-minute video with 10s chunks, expect ~30 chunks
    # For longer videos, transcription can take 10-30 minutes

    # === Image Configuration ===
    IMAGE_RESIZE_WIDTH: int = 1024
    IMAGE_RESIZE_HEIGHT: int = 768

    # === Retrieval Configuration ===
    TOP_K_IMAGE: int = 3
    TOP_K_AUDIO: int = 3
    TOP_K_DESCRIPTION: int = 3
    TOP_K_DOMAIN: int = 3
    FUSION_TOP_K: int = 10

    # === Fusion Weights ===
    WEIGHT_AUDIO: float = 0.3
    WEIGHT_IMAGE: float = 0.2
    WEIGHT_DESCRIPTION: float = 0.25
    WEIGHT_DOMAIN: float = 0.25

    # === API Configuration ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # === Data Paths ===
    DATA_DIR: str = "../data"
    VIDEO_DIR: str = "../data/videos"
    CACHE_DIR: str = "../data/cache"

    # === Prompt Templates ===
    DESCRIPTION_PROMPT: str = "Describe what is happening in this image in detail."
    DOMAIN_PROMPT_TEMPLATE: str = "Based on the context '{domain_context}', describe what you observe in this image."

    def get_video_dir(self) -> Path:
        """Get the video directory path."""
        return Path(self.VIDEO_DIR).resolve()

    def get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        return Path(self.CACHE_DIR).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


