"""FastAPI backend for QuadRAG."""

import asyncio
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# ============================================================================
# CRITICAL: Fix Python path and import numpy BEFORE any heavy imports
# This prevents the error: "you should not try to import numpy from its source directory"
# ============================================================================

# Step 1: Clean up sys.path - remove current directory and any potentially conflicting paths
_original_cwd = os.getcwd()
_paths_to_remove = []
for p in sys.path:
    # Remove current directory, empty strings, and any path containing 'numpy' source
    if p == '' or p == '.' or p == _original_cwd:
        _paths_to_remove.append(p)
    # Also check for paths that might be numpy source directories
    elif os.path.isdir(p):
        potential_numpy = os.path.join(p, 'numpy')
        if os.path.isdir(potential_numpy):
            setup_py = os.path.join(potential_numpy, 'setup.py')
            pyproject = os.path.join(potential_numpy, 'pyproject.toml')
            # If it looks like numpy source (has setup.py or pyproject.toml), remove it
            if os.path.exists(setup_py) or os.path.exists(pyproject):
                _paths_to_remove.append(p)

for p in _paths_to_remove:
    if p in sys.path:
        sys.path.remove(p)

# Step 2: Change to a safe working directory
if os.path.exists("/app"):
    os.chdir("/app")
elif os.path.exists("/tmp"):
    os.chdir("/tmp")

# Step 3: Import numpy from the correct location (installed package)
try:
    import numpy as np
    _numpy_path = getattr(np, '__file__', 'unknown')
    _numpy_version = getattr(np, '__version__', 'unknown')
except ImportError as e:
    _numpy_path = f"IMPORT_FAILED: {e}"
    _numpy_version = "N/A"

# Step 4: Configure event loop policy BEFORE any async imports
# Pixeltable uses nest_asyncio which can't patch uvloop event loops
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # Use standard asyncio, not uvloop
logger.info("Set asyncio event loop policy to DefaultEventLoopPolicy")

# Step 5: Defer Pixeltable initialization until needed to avoid asyncio conflicts
_pixeltable_initialized = False
def _ensure_pixeltable():
    global _pixeltable_initialized
    if not _pixeltable_initialized:
        try:
            import pixeltable as pxt
            pxt.init()  # Initialize Pixeltable when first needed
            _pixeltable_initialized = True
            logger.info("Pixeltable initialized successfully (lazy)")
        except Exception as e:
            logger.warning(f"Could not initialize Pixeltable: {e}")
            raise

# Step 6: Set up API keys for Pixeltable
try:
    from quadrag.config import get_settings
    _early_settings = get_settings()
    os.environ["OPENAI_API_KEY"] = _early_settings.OPENAI_API_KEY or ""
    os.environ["GOOGLE_API_KEY"] = _early_settings.GOOGLE_API_KEY or ""
    logger.info("API keys configured for Pixeltable")
except Exception as e:
    logger.warning(f"Could not configure API keys: {e}")

# Step 4: Restore working directory (but keep sys.path clean)
os.chdir(_original_cwd)

# Step 5: Re-add the original paths that were needed (but not the conflicting ones)
# Add backend/src to path for quadrag imports
_backend_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if _backend_src not in sys.path and os.path.isdir(_backend_src):
    sys.path.insert(0, _backend_src)
# ============================================================================

from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Log numpy import status
if 'IMPORT_FAILED' in str(_numpy_path):
    logger.error(f"NumPy import failed: {_numpy_path}")
    logger.error(f"Current sys.path: {sys.path}")
else:
    logger.info(f"NumPy {_numpy_version} pre-imported successfully from: {_numpy_path}")

# Import models first (lightweight)
from quadrag.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexType,
    ProcessingStatus,
    VideoListResponse,
    VideoMetadata,
    VideoProcessRequest,
    VideoProcessResponse,
    VideoStatusResponse,
    VideoUploadResponse,
)

# Thread-safe processing state (replaces the four ad-hoc module dicts).
# See backend/src/quadrag/state/processing_state.py for the full API.
from quadrag.state import ProcessingStateStore


def _resolve_snapshot_path() -> Path:
    """Work out where the ProcessingStateStore snapshot lives on this host."""
    from quadrag.config import get_settings
    return get_settings().get_cache_dir() / "processing_state.json"


def _on_fail_cleanup(video_id: str) -> None:
    """Cleanup callback invoked by ProcessingStateStore.mark_failed.

    Imports Pixeltable lazily so test-time imports of this module don't force
    a full Pixeltable init.
    """
    try:
        from quadrag.video.processor import cleanup_partial_pixeltable_artifacts
    except Exception:
        logger.exception("Could not import cleanup helper; skipping orphan cleanup")
        return
    cleanup_partial_pixeltable_artifacts(video_id)


# Instantiate the store with its snapshot path + cleanup hook. Contents are
# loaded from disk later via the FastAPI lifespan handler.
store = ProcessingStateStore(
    snapshot_path=_resolve_snapshot_path(),
    on_fail_cleanup=_on_fail_cleanup,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load persisted processing state on startup."""
    snapshot_path = _resolve_snapshot_path()
    restored = ProcessingStateStore.load_from_disk(
        snapshot_path,
        on_fail_cleanup=_on_fail_cleanup,
    )
    # Replace the module-level `store`'s contents in-place so any module that
    # already imported `api.store` keeps working.
    store.restore(restored.snapshot())
    logger.info(f"ProcessingStateStore restored from {snapshot_path}")
    yield


# Initialize FastAPI app with lifespan hook
app = FastAPI(
    title="QuadRAG API",
    description="A Four-Index Multimodal RAG System for Video Understanding",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Processing lock to prevent concurrent Pixeltable operations
processing_lock = asyncio.Lock()

# Lazy-loaded components (initialized on first use)
_settings = None
_video_processor = None
_indexer = None
_fusion = None
_generator = None
_search_engine = None
_initialized = False


def get_settings_lazy():
    """Lazy load settings."""
    global _settings
    if _settings is None:
        from quadrag.config import get_settings
        _settings = get_settings()
    return _settings


def get_video_processor():
    """Lazy load video processor."""
    global _video_processor
    if _video_processor is None:
        from quadrag.video.processor import VideoProcessor
        _video_processor = VideoProcessor()
    return _video_processor


def get_video_from_registry(video_id: str):
    """Lazy load get_video_from_registry function."""
    from quadrag.video.registry import get_video_from_registry
    return get_video_from_registry(video_id)


def get_all_videos():
    """Lazy load get_all_videos function."""
    from quadrag.video.registry import get_all_videos
    return get_all_videos()


def get_video_search_engine():
    """Lazy load VideoSearchEngine class."""
    from quadrag.retrieval.search_engine import VideoSearchEngine
    return VideoSearchEngine


def get_indexer_lazy():
    """Lazy load indexer."""
    global _indexer
    if _indexer is None:
        from quadrag.video.indexer import get_indexer
        _indexer = get_indexer()
    return _indexer


def get_fusion_lazy():
    """Lazy load fusion."""
    global _fusion
    if _fusion is None:
        from quadrag.retrieval.fusion import get_fusion
        _fusion = get_fusion()
    return _fusion


def get_generator_lazy():
    """Lazy load generator."""
    global _generator
    if _generator is None:
        from quadrag.generation.rag_generator import get_generator
        _generator = get_generator()
    return _generator


# Convenience aliases (will call lazy functions)
def _get_settings():
    return get_settings_lazy()

# Make settings available as module-level for compatibility
class _SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings_lazy(), name)

settings = _SettingsProxy()

logger.info("QuadRAG API module loaded (components will initialize on first use)")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="ok",
        message="QuadRAG API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        message="QuadRAG API is healthy"
    )


async def process_video_background(video_id: str, video_path: str):
    """Background task to transcode and process a video.

    Args:
        video_id: Video identifier
        video_path: Path to the uploaded video file
    """
    logger.info(f"Background task started for video {video_id}")
    try:
        logger.info(f"Starting background transcoding for video {video_id}")

        # Transcode the video first
        from quadrag.utils import validate_video_format, transcode_video, monitor_processing
        try:
            # Validate format for logging
            is_valid = validate_video_format(video_path)
            logger.info(f"Video validation result: {'PASSED' if is_valid else 'FAILED'}")

            # Always transcode to ensure pixeltable compatibility
            logger.info("Transcoding video to guaranteed compatible format (H.264 Main + AAC)...")

            # Monitor transcoding resource usage
            with monitor_processing("Video transcoding"):
                transcoded_path = transcode_video(video_path)

            # Clean up original file after successful transcoding
            from quadrag.utils import cleanup_processing_files
            cleanup_processing_files(video_path, transcoded_path)

            # Rename transcoded file to original location
            Path(transcoded_path).rename(video_path)
            logger.info("Video transcoded successfully to pixeltable-compatible format")

        except Exception as e:
            logger.error(f"Video transcoding failed for {video_id}: {e}")
            store.mark_failed(video_id, f"Transcoding failed: {str(e)}")
            return

        # Now start the actual video processing - run synchronously in separate thread
        logger.info(f"Starting video indexing for {video_id}")

        # Run Pixeltable operations in a daemon thread to avoid asyncio conflicts
        import threading
        def sync_processing():
            try:
                _process_video_sync(video_id, domain_context, session_id)
                logger.info(f"Background task completed successfully for video {video_id}")
            except Exception as e:
                logger.error(f"Background processing failed for {video_id}: {e}")
                import traceback
                logger.error(f"Background processing traceback: {traceback.format_exc()}")
                store.mark_failed(video_id, str(e))

        # Start as daemon thread - will not prevent server shutdown
        processing_thread = threading.Thread(target=sync_processing, daemon=True)
        processing_thread.start()

    except Exception as e:
        logger.error(f"Background processing failed for {video_id}: {e}")
        import traceback
        logger.error(f"Background processing traceback: {traceback.format_exc()}")
        store.mark_failed(video_id, str(e))


def _process_video_sync(video_id: str, domain_context: Optional[str] = None, session_id: Optional[str] = None):
    """Synchronous wrapper for video processing to avoid asyncio conflicts."""
    try:
        # Create a new event loop for Pixeltable operations
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_process_video_async(video_id, domain_context, session_id))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Sync video processing failed for {video_id}: {e}")
        import traceback
        logger.error(f"Sync processing traceback: {traceback.format_exc()}")
        store.mark_failed(video_id, str(e))


@app.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    domain_context: Optional[str] = Form(None, description="Optional domain context for domain-specific indexing"),
    session_id: Optional[str] = Form(None, description="Optional session ID for domain indexing")
):
    """Upload a video file.

    Args:
        file: Video file to upload
        domain_context: Optional domain context for domain-specific indexing
        session_id: Optional session ID for domain indexing

    Returns:
        VideoUploadResponse with video ID and file path
    """
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())

        # Ensure video directory exists
        video_dir = settings.get_video_dir()
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save video file
        file_extension = Path(file.filename).suffix
        file_path = video_dir / f"{video_id}{file_extension}"

        logger.info(f"Uploading video {video_id}: {file.filename}")

        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        # Clear extended attributes that might cause read issues (macOS specific)
        import subprocess
        try:
            subprocess.run(["xattr", "-c", str(file_path)], check=False, capture_output=True)
            logger.info("Cleared extended attributes from video file")
        except Exception as e:
            logger.warning(f"Could not clear extended attributes: {e}")
        
        # Quick validation before starting background processing
        from quadrag.utils import validate_video_size
        try:
            validate_video_size(str(file_path))
            logger.info("Video size validation passed")
        except Exception as e:
            logger.error(f"Video size validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Video validation failed: {str(e)}")

        # Start background processing (transcoding + indexing) asynchronously
        # This prevents the upload request from timing out on large videos
        store.mark_processing(video_id)
        logger.info(f"Starting background processing for video {video_id}")

        # Use asyncio.create_task in a fire-and-forget manner, but ensure proper exception handling
        task = asyncio.create_task(_process_video_background(video_id, domain_context, session_id))
        # Don't await the task - let it run in background
        # Remove callback to avoid event loop corruption during shutdown
        # The task will handle its own error logging internally

        logger.info(f"Video uploaded successfully: {file_path}")

        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            file_path=str(file_path),
            message="Video uploaded successfully",
        )

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """Process a video and create base indexes.

    Args:
        request: Video process request with video_id

    Returns:
        VideoProcessResponse with processing status
    """
    try:
        video_id = request.video_id

        if not store.has(video_id):
            raise HTTPException(status_code=404, detail="Video not found")

        # Check if already processed
        if get_video_processor().video_exists(video_id):
            return VideoProcessResponse(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                message="Video already processed",
            )

        # Start processing in background
        store.mark_processing(video_id)
        asyncio.create_task(_process_video_background(video_id, request.domain_context, request.session_id))

        return VideoProcessResponse(
            video_id=video_id,
            status=ProcessingStatus.PROCESSING,
            message="Video processing started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start processing: {str(e)}"
        )


@app.post("/reprocess-video", response_model=VideoProcessResponse)
async def reprocess_video(request: VideoProcessRequest):
    """Re-process a video to retry failed indexes.

    Args:
        request: Video process request with video_id

    Returns:
        VideoProcessResponse with processing status
    """
    try:
        video_id = request.video_id

        if not store.has(video_id):
            raise HTTPException(status_code=404, detail="Video not found")

        # Check if video exists in registry
        video_info = get_video_from_registry(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found in registry")

        # Reset processing status to allow re-processing; clear previous errors but
        # keep existing indexes so successful ones survive a retry.
        store.clear_errors(video_id)
        store.mark_processing(video_id)

        logger.info(f"Re-processing video {video_id}")

        # Start re-processing in background
        asyncio.create_task(_process_video_background(video_id, request.domain_context, request.session_id))

        return VideoProcessResponse(
            video_id=video_id,
            status=ProcessingStatus.PROCESSING,
            message="Video re-processing started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting video re-processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start re-processing: {str(e)}"
        )


async def _process_video_async(video_id: str, domain_context: Optional[str] = None, session_id: Optional[str] = None):
    """Process video and create indexes using thread isolation for Pixeltable operations.

    Args:
        video_id: Video identifier
        domain_context: Optional domain context for domain index
        session_id: Optional session ID for domain index
    """
    logger.info(f"Starting background processing for video {video_id}")

    # Get video file path
    video_dir = settings.get_video_dir()
    video_files = list(video_dir.glob(f"{video_id}*"))

    if not video_files:
        raise ValueError(f"Video file not found for ID: {video_id}")

    video_path = str(video_files[0])

    # Initialize result tracking
    indexes_created_list = []
    processing_error = None

    def _sync_pixeltable_processing():
        """Run Pixeltable operations in a separate thread with its own event loop."""
        nonlocal indexes_created_list, processing_error

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async Pixeltable operations in this thread's loop
            loop.run_until_complete(_run_pixeltable_ops_async(
                video_id, video_path, domain_context, session_id, indexes_created_list
            ))
        except Exception as e:
            processing_error = str(e)
            logger.error(f"Pixeltable processing failed for {video_id}: {e}")
        finally:
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop for {video_id}: {e}")

    async def _run_pixeltable_ops_async(video_id: str, video_path: str,
                                       domain_context: Optional[str], session_id: Optional[str],
                                       indexes_created_list: list):
        """Async wrapper for Pixeltable operations."""
        from quadrag.utils import monitor_processing

        with monitor_processing(f"Complete video processing for {video_id}"):
            # Process video (creates table and inserts video)
            logger.info(f"Processing video: {video_path}")
            get_video_processor().process_video(video_id, video_path)

        # Initialize index errors tracking for this video
        store.clear_errors(video_id)

        # Create Audio Index
        logger.info(f"Creating Audio Index for {video_id}")
        try:
            if get_indexer_lazy().create_audio_index(video_id):
                indexes_created_list.append(IndexType.AUDIO)
                logger.info(f"Audio Index created successfully for {video_id}")
            else:
                error_msg = "Audio index creation failed - audio format may be incompatible with transcription service"
                store.record_index_error(video_id, IndexType.AUDIO, error_msg)
                logger.warning(f"Audio Index creation failed for {video_id}: {error_msg}")
        except Exception as e:
            error_msg = f"Audio index creation error: {str(e)[:200]}"
            store.record_index_error(video_id, IndexType.AUDIO, error_msg)
            logger.error(f"Audio Index creation failed for {video_id}: {error_msg}")

        # Create Image Index
        logger.info(f"Creating Image Index for {video_id}")
        try:
            if get_indexer_lazy().create_image_index(video_id):
                indexes_created_list.append(IndexType.IMAGE)
                logger.info(f"Image Index created successfully for {video_id}")
            else:
                error_msg = "Image index creation failed - video frame extraction issue"
                store.record_index_error(video_id, IndexType.IMAGE, error_msg)
                logger.warning(f"Image Index creation failed for {video_id}: {error_msg}")
        except Exception as e:
            error_msg = f"Image index creation error: {str(e)[:200]}"
            store.record_index_error(video_id, IndexType.IMAGE, error_msg)
            logger.error(f"Image Index creation failed for {video_id}: {error_msg}")

        # Create Description Index (only if Image Index succeeded)
        if IndexType.IMAGE in indexes_created_list:
            logger.info(f"Creating Description Index for {video_id}")
            try:
                if get_indexer_lazy().create_description_index(video_id):
                    indexes_created_list.append(IndexType.DESCRIPTION)
                    logger.info(f"Description Index created successfully for {video_id}")
                else:
                    error_msg = "Description index creation failed - vision API issue"
                    store.record_index_error(video_id, IndexType.DESCRIPTION, error_msg)
                    logger.warning(f"Description Index creation failed for {video_id}: {error_msg}")
            except Exception as e:
                error_msg = f"Description index creation error: {str(e)[:200]}"
                store.record_index_error(video_id, IndexType.DESCRIPTION, error_msg)
                logger.error(f"Description Index creation failed for {video_id}: {error_msg}")
        else:
            logger.info(f"Skipping Description Index (requires Image Index)")

        # Create Domain Index eagerly if a context was supplied at upload time.
        # Without a context we skip — the lazy path in /chat builds per-query.
        if IndexType.IMAGE in indexes_created_list and domain_context:
            logger.info(f"Creating Domain Index for {video_id} with context: {domain_context}")
            try:
                if get_indexer_lazy().create_domain_index(video_id, domain_context):
                    indexes_created_list.append(IndexType.DOMAIN)
                    logger.info(f"Domain Index created successfully for {video_id}")
                else:
                    error_msg = "Domain index creation failed - vision API or context issue"
                    store.record_index_error(video_id, IndexType.DOMAIN, error_msg)
                    logger.warning(f"Domain Index creation failed for {video_id}: {error_msg}")
            except Exception as e:
                error_msg = f"Domain index creation error: {str(e)[:200]}"
                store.record_index_error(video_id, IndexType.DOMAIN, error_msg)
                logger.error(f"Domain Index creation failed for {video_id}: {error_msg}")
        else:
            if IndexType.IMAGE not in indexes_created_list:
                logger.info(f"Skipping Domain Index (requires Image Index)")
            else:
                logger.info(
                    f"No domain_context provided at upload time for {video_id}; "
                    f"domain view will be built lazily on first /chat request."
                )

    # Run Pixeltable operations in a separate daemon thread to avoid event loop corruption
    processing_thread = threading.Thread(target=_sync_pixeltable_processing, daemon=True)
    processing_thread.start()

    # Wait for the thread to complete (with timeout)
    processing_thread.join(timeout=1800)  # 30 minute timeout

    if processing_thread.is_alive():
        logger.error(f"Processing thread for {video_id} timed out after 30 minutes")
        store.mark_failed(video_id, "Processing timed out")
        return

    if processing_error:
        store.mark_failed(video_id, processing_error)
        logger.error(f"Processing failed for {video_id}: {processing_error}")
        return

    # Mark as completed even if some indexes failed (partial success)
    if indexes_created_list:
        store.mark_completed(video_id, indexes_created_list)
        logger.info(f"Successfully processed video {video_id} with indexes: {indexes_created_list}")
        logger.info(f"Status updated to COMPLETED for video {video_id}, current status: {store.get_status(video_id)}")
    else:
        store.mark_failed(video_id, "All index creation failed")
        logger.error(f"Failed to create any indexes for video {video_id}")




async def _process_video_background(video_id: str, domain_context: Optional[str] = None, session_id: Optional[str] = None):
    """Background task to process video and create indexes.

    Args:
        video_id: Video identifier
        domain_context: Optional domain context for domain index
        session_id: Optional session ID for domain index
    """
    # Use processing lock to prevent concurrent Pixeltable operations
    async with processing_lock:
        # Run Pixeltable operations directly in async context
        try:
            await _process_video_async(video_id, domain_context, session_id)
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            store.mark_failed(video_id, str(e))
            raise


@app.get("/video/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(video_id: str):
    """Get video processing status.

    Args:
        video_id: Video identifier

    Returns:
        VideoStatusResponse with status and created indexes
    """
    try:
        status = store.get_status(video_id)
        error_message = store.get_error(video_id)
        indexes_created = store.get_indexes(video_id)

        logger.debug(f"Video {video_id} status: {status}, indexes: {indexes_created}")
        
        # Also verify from registry if available
        video_info = get_video_from_registry(video_id)
        if video_info and not indexes_created:
            # Fallback: check which indexes exist in registry
            try:
                if video_info.frames_view:
                    if IndexType.IMAGE not in indexes_created:
                        indexes_created.append(IndexType.IMAGE)
                    if IndexType.DESCRIPTION not in indexes_created:
                        indexes_created.append(IndexType.DESCRIPTION)
            except Exception:
                logger.exception(f"Failed to probe frames_view for {video_id}")

            try:
                if video_info.audio_view:
                    if IndexType.AUDIO not in indexes_created:
                        indexes_created.append(IndexType.AUDIO)
            except Exception:
                logger.exception(f"Failed to probe audio_view for {video_id}")

            try:
                # Any registered domain view counts for the IndexType.DOMAIN flag;
                # post-Step-8 there can be several per video keyed by context hash.
                if video_info.domain_views:
                    if IndexType.DOMAIN not in indexes_created:
                        indexes_created.append(IndexType.DOMAIN)
            except Exception:
                logger.exception(f"Failed to probe domain_views for {video_id}")

        logger.debug(f"Video {video_id} status: {status}, indexes: {indexes_created}")
        return VideoStatusResponse(
            video_id=video_id,
            status=status,
            indexes_created=indexes_created,
            error_message=error_message,
            index_errors=store.get_index_errors(video_id),
        )

    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the video using RAG.

    Args:
        request: Chat request with query

    Returns:
        ChatResponse with answer and citations
    """
    try:
        logger.info(f"Processing chat request for video {request.video_id}")

        # Check if video exists
        if not get_video_processor().video_exists(request.video_id):
            raise HTTPException(status_code=404, detail="Video not found or not processed")

        # Run search directly in async context since Pixeltable needs proper event loop
        try:
            # If a domain_context was supplied, make sure the per-context domain view
            # exists before we search — lazy-building it here (may take 30s–2min on
            # a first-time context) is the Step-8 design.
            domain_view_name: Optional[str] = None
            if request.domain_context:
                from quadrag.video.domain_manager import ensure_domain_view
                domain_view_name = ensure_domain_view(request.video_id, request.domain_context)

            # Initialize search engine directly
            from quadrag.retrieval.search_engine import VideoSearchEngine
            search_engine = VideoSearchEngine(
                request.video_id,
                request.session_id,
                domain_view_name=domain_view_name,
            )

            # Search all indexes
            search_results = search_engine.search_all_indexes(
                query_text=request.query,
                use_domain=domain_view_name is not None,
            )

            # Debug: Log search results
            total_results = sum(len(results) for results in search_results.values())
            logger.debug(f"Search completed - total results: {total_results}")
            for index_type, results in search_results.items():
                logger.debug(f"{index_type.value}: {len(results)} results")
                if results:
                    for i, result in enumerate(results[:2]):  # Show first 2 results
                        logger.debug(f"  Result {i}: '{result.content[:100]}...' at {result.timestamp:.1f}s (score: {result.similarity:.3f})")

            # Fuse results
            from quadrag.retrieval.fusion import get_fusion
            fusion = get_fusion()
            fused_results = fusion.fuse_results(search_results)

            logger.debug(f"Fused results: {len(fused_results)}")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            fused_results = []

        # Generate answer (this doesn't use Pixeltable, so it's fine to run async)
        response = get_generator_lazy().generate_answer(
            query=request.query,
            retrieved_results=fused_results,
            domain_context=request.domain_context,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos", response_model=VideoListResponse)
async def list_videos():
    """List all processed videos.

    Returns:
        VideoListResponse with list of videos
    """
    try:
        all_videos = get_all_videos()
        
        video_metadata_list = []
        for video_id, video_info in all_videos.items():
            status = store.get_status(video_id, default=ProcessingStatus.COMPLETED)
            
            # Get created indexes
            indexes_created = []
            try:
                if video_info.frames_view:
                    indexes_created.extend([IndexType.IMAGE, IndexType.DESCRIPTION])
                if video_info.audio_view:
                    indexes_created.append(IndexType.AUDIO)
                if video_info.domain_views:
                    indexes_created.append(IndexType.DOMAIN)
            except Exception:
                logger.exception(f"Failed to enumerate indexes for {video_id}")

            # Get video file info
            video_dir = settings.get_video_dir()
            video_files = list(video_dir.glob(f"{video_id}*"))
            filename = video_files[0].name if video_files else "unknown"

            metadata = VideoMetadata(
                video_id=video_id,
                filename=filename,
                upload_time=datetime.now(),  # Would need to store this
                status=status,
                indexes_created=indexes_created,
            )
            video_metadata_list.append(metadata)

        return VideoListResponse(videos=video_metadata_list)

    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check all video processing status."""
    return {
        "store": store.snapshot(),
        "registry_videos": list(get_all_videos().keys()) if get_all_videos else [],
    }


@app.get("/debug/transcriptions/{video_id}")
async def debug_transcriptions(video_id: str):
    """Debug endpoint to force computation and return transcriptions."""
    try:
        from quadrag.retrieval.search_engine import VideoSearchEngine

        # Initialize search engine
        VideoSearchEngineClass = get_video_search_engine()
        search_engine = VideoSearchEngineClass(video_id, None)

        # Force transcription computation and return results
        if hasattr(search_engine.video_info, 'audio_view') and search_engine.video_info.audio_view:
            audio_view = search_engine.video_info.audio_view
            try:
                # Force computation by collecting all transcriptions
                chunks = audio_view.select(
                    audio_view.segment_start,
                    audio_view.segment_end,
                    audio_view.transcript_text,
                ).collect()

                transcriptions = []
                for i, chunk in enumerate(chunks):
                    text = str(chunk.get("transcript_text", "")).strip()
                    raw_transcript = chunk.get("transcription", "")
                    logger.info(f"Debug chunk {i}: text='{text[:100]}...', raw='{str(raw_transcript)[:200]}...'")
                    transcriptions.append({
                        "start_time": float(chunk.get("segment_start", 0)),
                        "end_time": float(chunk.get("segment_end", 0)),
                        "text": text,
                        "raw_transcription": str(raw_transcript)
                    })

                return {
                    "video_id": video_id,
                    "transcription_count": len(transcriptions),
                    "transcriptions": transcriptions
                }
            except Exception as e:
                return {"error": f"Failed to collect transcriptions: {e}"}
        else:
            return {"error": "No audio view available for this video"}

    except Exception as e:
        logger.error(f"Debug transcriptions error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import os
    import sys
    import uvicorn
    
    try:
        # Get settings (now with defaults, won't crash)
        _settings = get_settings_lazy()
        
        # Ensure data directories exist
        video_dir = _settings.get_video_dir()
        cache_dir = _settings.get_cache_dir()
        video_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directories ready: {video_dir}, {cache_dir}")
        
        # Check API keys (warn but don't exit - let healthcheck pass)
        missing_keys = []
        if not _settings.GROQ_API_KEY:
            missing_keys.append("GROQ_API_KEY")
        if not _settings.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not _settings.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")
        
        if missing_keys:
            logger.warning(f"Missing API keys: {missing_keys}. Some features will not work.")
            logger.warning("Set these in Railway environment variables.")
        else:
            logger.info("All API keys configured")
        
        # Use PORT from environment (Railway) or fall back to API_PORT
        port = int(os.environ.get("PORT", _settings.API_PORT))
        host = _settings.API_HOST
        
        logger.info(f"Starting QuadRAG API on {host}:{port}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            loop="asyncio",  # Use standard asyncio, not uvloop
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


