"""FastAPI backend for QuadRAG."""

import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
print("INFO: Set asyncio event loop policy to DefaultEventLoopPolicy")

# Step 5: Initialize Pixeltable with standard asyncio event loop
try:
    import pixeltable as pxt
    pxt.init()  # Initialize Pixeltable with standard asyncio event loop
    print("INFO: Pixeltable initialized successfully")
except Exception as e:
    print(f"WARNING: Could not initialize Pixeltable: {e}")

# Step 6: Set up API keys for Pixeltable
try:
    from quadrag.config import get_settings
    _early_settings = get_settings()
    os.environ["OPENAI_API_KEY"] = _early_settings.OPENAI_API_KEY or ""
    os.environ["GOOGLE_API_KEY"] = _early_settings.GOOGLE_API_KEY or ""
    print("INFO: API keys configured for Pixeltable")
except Exception as e:
    print(f"WARNING: Could not configure API keys: {e}")

# Step 4: Restore working directory (but keep sys.path clean)
os.chdir(_original_cwd)

# Step 5: Re-add the original paths that were needed (but not the conflicting ones)
# Add backend/src to path for quadrag imports
_backend_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if _backend_src not in sys.path and os.path.isdir(_backend_src):
    sys.path.insert(0, _backend_src)
# ============================================================================

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

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
    DomainContextRequest,
    DomainContextResponse,
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

# Initialize FastAPI app FIRST (before any heavy imports)
app = FastAPI(
    title="QuadRAG API",
    description="A Four-Index Multimodal RAG System for Video Understanding",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor for Pixeltable operations
executor = ThreadPoolExecutor(max_workers=2)

# Track processing status
processing_status: Dict[str, ProcessingStatus] = {}
processing_errors: Dict[str, str] = {}
video_indexes: Dict[str, list[IndexType]] = {}

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


@app.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file.

    Args:
        file: Video file to upload

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
        
        # Always transcode video to ensure compatibility with pixeltable/ffmpeg
        from quadrag.utils import validate_video_format, transcode_video
        try:
            # Validate first for logging
            is_valid = validate_video_format(str(file_path))
            logger.info(f"Video validation result: {'PASSED' if is_valid else 'FAILED'}")

            # Always transcode to ensure pixeltable compatibility
            logger.info("Transcoding video to guaranteed compatible format (H.264 Main + AAC)...")
            transcoded_path = transcode_video(str(file_path))
            # Replace original with transcoded
            file_path.unlink()
            Path(transcoded_path).rename(file_path)
            logger.info("Video transcoded successfully to pixeltable-compatible format")
        except Exception as e:
            logger.error(f"Video transcoding failed: {e}")
            raise HTTPException(status_code=400, detail=f"Video processing failed: {str(e)}")

        processing_status[video_id] = ProcessingStatus.PENDING

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

        if video_id not in processing_status:
            raise HTTPException(status_code=404, detail="Video not found")

        # Check if already processed
        if get_video_processor().video_exists(video_id):
            return VideoProcessResponse(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                message="Video already processed",
            )

        # Start processing in background
        processing_status[video_id] = ProcessingStatus.PROCESSING
        asyncio.create_task(_process_video_background(video_id))

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


async def _process_video_async(video_id: str):
    """Async video processing function (runs in event loop).

    Args:
        video_id: Video identifier
    """
    logger.info(f"Starting background processing for video {video_id}")

    # Get video file path
    video_dir = settings.get_video_dir()
    video_files = list(video_dir.glob(f"{video_id}*"))

    if not video_files:
        raise ValueError(f"Video file not found for ID: {video_id}")

    video_path = str(video_files[0])

    # Wrap Pixeltable operations in a task to ensure proper async context
    indexes_created_list = []
    
    async def _run_pixeltable_ops():
        """Run Pixeltable operations in a proper async task context."""
        nonlocal indexes_created_list
        
        # Process video (creates table and inserts video)
        logger.info(f"Processing video: {video_path}")
        get_video_processor().process_video(video_id, video_path)

        # Create Audio Index (prioritize this over image index due to PyTorch compatibility issues)
        logger.info(f"Creating Audio Index for {video_id}")
        if get_indexer_lazy().create_audio_index(video_id):
            indexes_created_list.append(IndexType.AUDIO)
            logger.info(f"Audio Index created successfully for {video_id}")
        else:
            logger.warning(f"Audio Index creation failed for {video_id}")

        # Create Image Index
        logger.info(f"Creating Image Index for {video_id}")
        if get_indexer_lazy().create_image_index(video_id):
            indexes_created_list.append(IndexType.IMAGE)
            logger.info(f"Image Index created successfully for {video_id}")
        else:
            logger.warning(f"Image Index creation failed for {video_id}")

        # Create Description Index (only if Image Index succeeded, as it depends on resized_frame)
        if IndexType.IMAGE in indexes_created_list:
            logger.info(f"Creating Description Index for {video_id}")
            if get_indexer_lazy().create_description_index(video_id):
                indexes_created_list.append(IndexType.DESCRIPTION)
                logger.info(f"Description Index created successfully for {video_id}")
            else:
                logger.warning(f"Description Index creation failed for {video_id}")
        else:
            logger.info(f"Skipping Description Index (requires Image Index)")

    # Run Pixeltable operations as a task to ensure proper context
    task = asyncio.create_task(_run_pixeltable_ops())
    await task

    # Mark as completed even if some indexes failed (partial success)
    if indexes_created_list:
        processing_status[video_id] = ProcessingStatus.COMPLETED
        video_indexes[video_id] = indexes_created_list
        logger.info(f"Successfully processed video {video_id} with indexes: {indexes_created_list}")
    else:
        processing_status[video_id] = ProcessingStatus.FAILED
        processing_errors[video_id] = "All index creation failed"
        video_indexes[video_id] = []
        logger.error(f"Failed to create any indexes for video {video_id}")




async def _process_video_background(video_id: str):
    """Background task to process video and create indexes.

    Args:
        video_id: Video identifier
    """
    # Use processing lock to prevent concurrent Pixeltable operations
    async with processing_lock:
        # Run Pixeltable operations directly in async context
        try:
            await _process_video_async(video_id)
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            processing_status[video_id] = ProcessingStatus.FAILED
            processing_errors[video_id] = str(e)
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
        status = processing_status.get(video_id, ProcessingStatus.PENDING)
        error_message = processing_errors.get(video_id)

        # Get created indexes from our tracking
        indexes_created = video_indexes.get(video_id, [])
        
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
            except:
                pass

            try:
                if video_info.audio_view:
                    if IndexType.AUDIO not in indexes_created:
                        indexes_created.append(IndexType.AUDIO)
            except:
                pass

            try:
                if video_info.domain_view:
                    if IndexType.DOMAIN not in indexes_created:
                        indexes_created.append(IndexType.DOMAIN)
            except:
                pass

        logger.debug(f"Video {video_id} status: {status}, indexes: {indexes_created}")
        return VideoStatusResponse(
            video_id=video_id,
            status=status,
            indexes_created=indexes_created,
            error_message=error_message,
        )

    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set-domain-context", response_model=DomainContextResponse)
async def set_domain_context(request: DomainContextRequest):
    """Set domain context for a session and create domain index.

    Args:
        request: Domain context request

    Returns:
        DomainContextResponse with status
    """
    try:
        logger.info(
            f"Setting domain context for video {request.video_id}, "
            f"session {request.session_id}: {request.domain_context}"
        )

        # Check if video exists
        if not get_video_processor().video_exists(request.video_id):
            raise HTTPException(status_code=404, detail="Video not found")

        # Create domain index in thread pool (with new event loop)
        async def _create_domain_index_async(video_id: str, session_id: str, domain_context: str):
            """Create domain index async function."""
            # Wrap in a task to ensure proper async context
            async def _run_index_creation():
                return get_indexer_lazy().create_domain_index(video_id, session_id, domain_context)
            
            task = asyncio.create_task(_run_index_creation())
            return await task
        
        def _create_domain_index_sync(video_id: str, session_id: str, domain_context: str):
            """Create domain index with new event loop."""
            # Use asyncio.run() which properly sets up the event loop and task context
            return asyncio.run(_create_domain_index_async(video_id, session_id, domain_context))
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            executor,
            _create_domain_index_sync,
            request.video_id,
            request.session_id,
            request.domain_context
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create domain index")

        return DomainContextResponse(
            session_id=request.session_id,
            video_id=request.video_id,
            domain_context=request.domain_context,
            status="success",
            message="Domain context set successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting domain context: {e}")
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
            # Initialize search engine directly
            from quadrag.retrieval.search_engine import VideoSearchEngine
            search_engine = VideoSearchEngine(request.video_id, request.session_id)

            # Search all indexes
            search_results = search_engine.search_all_indexes(
                query_text=request.query,
                use_domain=request.domain_context is not None,
            )

            # Debug: Log search results
            total_results = sum(len(results) for results in search_results.values())
            print(f"DEBUG: Search completed - total results: {total_results}")
            for index_type, results in search_results.items():
                print(f"DEBUG: {index_type.value}: {len(results)} results")
                if results:
                    for i, result in enumerate(results[:2]):  # Show first 2 results
                        print(f"DEBUG:   Result {i}: '{result.content[:100]}...' at {result.timestamp:.1f}s (score: {result.similarity:.3f})")

            # Fuse results
            from quadrag.retrieval.fusion import get_fusion
            fusion = get_fusion()
            fused_results = fusion.fuse_results(search_results)

            print(f"DEBUG: Fused results: {len(fused_results)}")

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
            status = processing_status.get(video_id, ProcessingStatus.COMPLETED)
            
            # Get created indexes
            indexes_created = []
            try:
                if video_info.frames_view:
                    indexes_created.extend([IndexType.IMAGE, IndexType.DESCRIPTION])
                if video_info.audio_view:
                    indexes_created.append(IndexType.AUDIO)
                if video_info.domain_view:
                    indexes_created.append(IndexType.DOMAIN)
            except:
                pass

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
        "processing_status": {k: str(v) for k, v in processing_status.items()},
        "video_indexes": {k: [str(idx) for idx in v] for k, v in video_indexes.items()},
        "processing_errors": processing_errors,
        "registry_videos": list(get_all_videos().keys()) if get_all_videos else []
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
                    audio_view.start_time_sec,
                    audio_view.end_time_sec,
                    audio_view.transcript_text,
                ).collect()

                transcriptions = []
                for i, chunk in enumerate(chunks):
                    text = str(chunk.get("transcript_text", "")).strip()
                    raw_transcript = chunk.get("transcription", "")
                    logger.info(f"Debug chunk {i}: text='{text[:100]}...', raw='{str(raw_transcript)[:200]}...'")
                    transcriptions.append({
                        "start_time": float(chunk.get("start_time_sec", 0)),
                        "end_time": float(chunk.get("end_time_sec", 0)),
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


