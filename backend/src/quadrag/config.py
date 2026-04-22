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

    # === API Keys (set via environment variables) ===
    # OpenAI is still required for Whisper (audio index) and text-embedding-3-small
    # (all semantic search indexes). OpenRouter handles chat answers + frame
    # vision captions — neither of which OpenAI's Tier-1 TPM limits work for.
    OPENAI_API_KEY: str = ""        # Whisper + embeddings. Keep.
    OPENROUTER_API_KEY: str = ""    # Chat answers + vision captions.
    GOOGLE_API_KEY: str = ""        # Legacy — unused by active code.
    GROQ_API_KEY: str = ""          # Legacy — Groq was replaced by OpenRouter; env var accepted for .env compat.

    # === Model Configuration ===
    AUDIO_TRANSCRIPT_MODEL: str = "whisper-1"                        # OpenAI
    TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"             # OpenAI
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"      # Local CLIP
    # OpenRouter routes below. Change via env to try other providers.
    IMAGE_CAPTION_MODEL: str = "google/gemini-2.0-flash-001"         # OpenRouter — vision
    CHAT_MODEL: str = "meta-llama/llama-3.3-70b-instruct"            # OpenRouter — chat answer
    # Legacy alias; still read by rag_generator until callers are migrated.
    GROQ_MODEL: str = "meta-llama/llama-3.3-70b-instruct"

    # OpenRouter base URL. Override to point at a private proxy if desired.
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # === Video Processing Configuration ===
    SPLIT_FRAMES_COUNT: int = 20  # Cost-safe default; calculate_frame_count overrides per-duration
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

    # Time window (seconds) inside which two results are treated as duplicates.
    FUSION_DEDUP_WINDOW_SEC: float = 2.0

    # === LLM / Vision Generation Parameters ===
    # Groq chat completion (answer generation)
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 1024
    # OpenAI vision (frame description / domain caption UDFs)
    VISION_TEMPERATURE: float = 0.3
    VISION_MAX_TOKENS: int = 200

    # === Domain Index ===
    # Upper bound on distinct domain contexts kept as Pixeltable views per
    # video. Exceeding this triggers LRU eviction of the least-recently-used
    # view — both the registry entry and the Pixeltable view are dropped.
    MAX_DOMAIN_VIEWS_PER_VIDEO: int = 5
    # The tuning knobs below are legacy (difflib + manual-loop path, pre-Step-7).
    # They are unreferenced in production code now; keeping them only so old
    # .env files don't error out on extra-field rejection.
    DOMAIN_CAPTION_BATCH_SIZE: int = 5
    DOMAIN_CAPTION_BATCH_SLEEP_SEC: float = 2.0
    DOMAIN_SIMILARITY_THRESHOLD: float = 0.01
    DOMAIN_SEQUENCE_WEIGHT: float = 0.7
    DOMAIN_WORD_OVERLAP_WEIGHT: float = 0.3

    # === Client Timeouts (seconds) ===
    UPLOAD_TIMEOUT_SEC: int = 60
    CHAT_TIMEOUT_SEC: int = 120
    STATUS_POLL_TIMEOUT_SEC: int = 30

    # === Citation Grounding (used in a later step) ===
    CITATION_TIMESTAMP_TOLERANCE_SEC: float = 3.0

    # === API Configuration ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    PORT: int = 8000  # Railway sets this automatically
    
    # === Database Configuration (for Railway PostgreSQL) ===
    DATABASE_URL: str = ""  # Railway PostgreSQL URL (optional)
    PIXELTABLE_HOME: str = ""  # Custom Pixeltable home directory

    # === Data Paths ===
    DATA_DIR: str = "../data"
    VIDEO_DIR: str = "../data/videos"
    CACHE_DIR: str = "../data/cache"
    
    def get_api_port(self) -> int:
        """Get API port - uses Railway's PORT if set, otherwise API_PORT."""
        import os
        return int(os.environ.get("PORT", self.API_PORT))
    
    def get_video_dir(self) -> Path:
        """Get the video directory path."""
        import os
        # In Railway, use absolute path from /app
        if os.environ.get("RAILWAY_ENVIRONMENT"):
            return Path("/app/data/videos").resolve()
        return Path(self.VIDEO_DIR).resolve()

    def get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        import os
        # In Railway, use absolute path from /app
        if os.environ.get("RAILWAY_ENVIRONMENT"):
            return Path("/app/data/cache").resolve()
        return Path(self.CACHE_DIR).resolve()

    # === Prompt Templates ===
    DESCRIPTION_PROMPT: str = "Describe what is happening in this image in detail."
    DOMAIN_PROMPT_TEMPLATE: str = "Based on the context '{domain_context}', describe what you observe in this image."


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


