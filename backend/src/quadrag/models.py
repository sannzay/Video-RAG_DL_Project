"""Pydantic models for QuadRAG API."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Video processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexType(str, Enum):
    """Type of index."""

    IMAGE = "image"
    AUDIO = "audio"
    DESCRIPTION = "description"
    DOMAIN = "domain"


class VideoUploadResponse(BaseModel):
    """Response after video upload."""

    video_id: str
    filename: str
    file_path: str
    message: str


class VideoProcessRequest(BaseModel):
    """Request to process a video."""

    video_id: str
    domain_context: Optional[str] = None
    session_id: Optional[str] = None


class VideoProcessResponse(BaseModel):
    """Response after starting video processing."""

    video_id: str
    status: ProcessingStatus
    message: str


class VideoStatusResponse(BaseModel):
    """Video processing status response."""

    video_id: str
    status: ProcessingStatus
    indexes_created: list[IndexType]
    error_message: Optional[str] = None
    index_errors: dict[IndexType, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    content: str
    timestamp: float
    similarity: float
    source: IndexType


class ChatRequest(BaseModel):
    """Chat request with optional domain context."""

    session_id: str
    video_id: str
    query: str
    domain_context: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response with answer and citations.

    ``grounded`` is True when the LLM answer cited at least one ``[M:SS]``
    timestamp that matched one of the retrieved chunks (within the
    ``CITATION_TIMESTAMP_TOLERANCE_SEC`` window). False otherwise — the answer
    might still be useful but the UI should hint to the user that it isn't
    anchored to specific moments in the video.
    """

    answer: str
    citations: list[RetrievalResult]
    processing_time: float
    grounded: bool = False


class VideoMetadata(BaseModel):
    """Video metadata."""

    video_id: str
    filename: str
    upload_time: datetime
    status: ProcessingStatus
    duration: Optional[float] = None
    indexes_created: list[IndexType] = Field(default_factory=list)


class VideoListResponse(BaseModel):
    """Response with list of videos."""

    videos: list[VideoMetadata]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    message: str


