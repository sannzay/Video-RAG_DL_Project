"""Video indexer for creating the four semantic indexes."""

import os
from typing import Optional

import pixeltable as pxt
from loguru import logger
from pixeltable.functions import openai as pxt_openai
from pixeltable.functions.huggingface import clip
from pixeltable.functions.openai import embeddings
from pixeltable.functions.video import extract_audio
from pixeltable.functions.video import legacy_frame_iterator

from quadrag.config import get_settings
from quadrag.utils import calculate_frame_count, monitor_processing
from quadrag.video.functions import extract_text_from_chunk, resize_image
from quadrag.video.registry import (
    add_domain_view,
    build_domain_view_name,
    get_video_from_registry,
    hash_domain_context,
)

settings = get_settings()
logger = logger.bind(name="VideoIndexer")


class VideoIndexer:
    """Creates and manages the four semantic indexes for videos."""

    def __init__(self):
        """Initialize the video indexer."""
        self.settings = settings
        # Ensure the OpenAI key is in the environment for Pixeltable's async UDFs
        # (pxt_openai.vision / embeddings / transcriptions) to pick up. They read
        # it lazily via openai.OpenAI() internally; we just need to make sure it's set.
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    def create_image_index(self, video_id: str) -> bool:
        """Create Image Index with CLIP embeddings.

        Args:
            video_id: Video identifier

        Returns:
            True if successful
        """
        # Ensure Pixeltable is initialized before any operations
        try:
            from ...api import _ensure_pixeltable
            _ensure_pixeltable()
        except ImportError:
            # If we're not in the API context, Pixeltable should already be initialized
            pass

        try:
            logger.info(f"Creating Image Index for video {video_id}")
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            video_table = video_info.video_table

            # Get video duration for adaptive frame sampling
            video_duration = 0.0
            try:
                # Try to get duration from video table metadata if available
                # Otherwise use utility function (this might be expensive for large videos)
                from quadrag.utils import get_video_duration
                video_path = str(video_table.video)
                video_duration = get_video_duration(video_path)
                logger.info(f"Video duration: {video_duration:.1f} seconds ({video_duration/3600:.2f} hours)")
            except Exception as e:
                logger.warning(f"Could not determine video duration: {e}, using default frame count")

            # Calculate optimal frame count based on video duration
            optimal_frame_count = calculate_frame_count(video_duration)
            logger.info(f"Using adaptive frame sampling: {optimal_frame_count} frames for {video_duration:.1f}s video")

            # Create frames view with adaptive sampling
            logger.info(f"Creating frames view: {video_info.frames_view_name}")
            try:
                logger.info(f"Attempting to create frame iterator with {optimal_frame_count} frames")
                logger.info(f"Video file: {video_table.video}")
                frames_view = pxt.create_view(
                    video_info.frames_view_name,
                    video_table,
                    iterator=legacy_frame_iterator(
                        video=video_table.video,
                        num_frames=optimal_frame_count,
                    ),
                    if_exists="replace_force",
                )
            except Exception as e:
                logger.error(f"Failed to create frame iterator with {optimal_frame_count} frames: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Try with fewer frames as fallback
                fallback_frames = min(30, optimal_frame_count // 2)
                logger.info(f"Retrying with {fallback_frames} frames...")
                try:
                    frames_view = pxt.create_view(
                        video_info.frames_view_name,
                        video_table,
                        iterator=legacy_frame_iterator(
                            video=video_table.video,
                            num_frames=fallback_frames,
                        ),
                        if_exists="replace_force",
                    )
                except Exception as e2:
                    logger.error(f"Failed even with {fallback_frames} frames: {e2}")
                    logger.error(f"Exception type: {type(e2).__name__}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise e2

            # Add resized frame column using custom UDF
            frames_view.add_computed_column(
                resized_frame=resize_image(
                    frames_view.frame,
                    width=settings.IMAGE_RESIZE_WIDTH,
                    height=settings.IMAGE_RESIZE_HEIGHT,
                ),
                if_exists="ignore",
            )

            # Add CLIP embedding index
            logger.info("Creating CLIP embedding index")
            frames_view.add_embedding_index(
                column=frames_view.resized_frame,
                image_embed=clip.using(model_id=settings.IMAGE_EMBEDDING_MODEL),
                if_exists="replace_force",
            )

            logger.info(f"Successfully created Image Index for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Image Index: {e}")
            return False

    def create_audio_index(self, video_id: str) -> bool:
        """Create Audio Index with transcription and text embeddings.

        Args:
            video_id: Video identifier

        Returns:
            True if successful
        """
        # Ensure Pixeltable is initialized before any operations
        try:
            from ...api import _ensure_pixeltable
            _ensure_pixeltable()
        except ImportError:
            # If we're not in the API context, Pixeltable should already be initialized
            pass

        try:
            logger.info(f"Creating Audio Index for video {video_id}")
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            video_table = video_info.video_table

            # Extract audio
            logger.info("Extracting audio from video")
            video_table.add_computed_column(
                audio_extract=extract_audio(video_table.video, format="wav"),  # Use WAV for better Whisper compatibility
                if_exists="ignore",
            )

            # Create audio chunks view
            logger.info(f"Creating audio chunks view: {video_info.audio_view_name}")
            # Pixeltable 0.5.x renamed AudioSplitter → audio_splitter and reshaped
            # its params: chunk_duration_sec → duration, overlap_sec → overlap,
            # min_chunk_duration_sec → min_segment_duration. The output columns
            # also moved: audio_chunk → audio_segment, start_time_sec →
            # segment_start, end_time_sec → segment_end.
            from pixeltable.functions.audio import audio_splitter
            audio_view = pxt.create_view(
                video_info.audio_view_name,
                video_table,
                iterator=audio_splitter(
                    video_table.audio_extract,
                    duration=settings.AUDIO_CHUNK_LENGTH,
                    overlap=settings.AUDIO_OVERLAP_SECONDS,
                    min_segment_duration=settings.AUDIO_MIN_CHUNK_DURATION_SECONDS,
                ),
                if_exists="replace_force",
            )
            
            # Log chunk count for progress tracking
            chunk_count = len(audio_view.select().collect())
            logger.info(f"Created {chunk_count} audio chunks for transcription")

            # Add transcription column
            # IMPORTANT: We don't force evaluation here - transcriptions will compute lazily
            logger.info(f"Adding audio transcription column for {chunk_count} chunks")
            logger.info("Note: Transcriptions will compute lazily when queried. No blocking computation here.")
            
            # Check if column already exists to avoid re-adding
            try:
                # Try to access the column - if it exists, skip adding
                _ = audio_view.transcription
                logger.info("Transcription column already exists, skipping addition")
            except AttributeError:
                # Column doesn't exist, add it
                logger.info("Adding new transcription column...")
                logger.debug(f"About to add transcription column for {chunk_count} chunks")
                try:
                    audio_view.add_computed_column(
                        transcription=pxt_openai.transcriptions(
                            audio=audio_view.audio_segment,
                            model=settings.AUDIO_TRANSCRIPT_MODEL,
                        ),
                        if_exists="ignore",
                    )
                    logger.debug("Transcription column added successfully")
                    logger.info("Transcription column added successfully (will compute on-demand)")
                except Exception as e:
                    logger.error(f"Failed to add transcription column: {e}")
                    raise

            # Extract text from transcription
            # Check if column already exists
            try:
                _ = audio_view.transcript_text
                logger.info("Transcript text column already exists, skipping addition")
            except AttributeError:
                logger.info("Adding text extraction column...")
                logger.debug("About to add text extraction column")
                try:
                    audio_view.add_computed_column(
                        transcript_text=extract_text_from_chunk(audio_view.transcription),
                        if_exists="ignore",
                    )
                    logger.debug("Text extraction column added successfully")
                    logger.info("Text extraction column added successfully (will compute on-demand)")
                except Exception as e:
                    logger.error(f"Failed to add text extraction column: {e}")
                    raise

            # Force Whisper to run on every chunk before we build the embedding index.
            # Without this, add_embedding_index would trigger materialization as a
            # side-effect, which can race with other query paths and swallows errors.
            # Calling .collect() up-front makes Whisper failures loud and catchable.
            logger.info("Pre-computing Whisper transcriptions for audio chunks")
            audio_view.select(
                audio_view.transcription,
                audio_view.transcript_text,
            ).collect()
            logger.info("All transcriptions materialized")

            # Build the semantic search index over the transcript text. Same pattern
            # as create_description_index — relies on the transcript_text column
            # already being populated by the .collect() above.
            logger.info("Creating text embedding index on transcript_text")
            audio_view.add_embedding_index(
                column=audio_view.transcript_text,
                string_embed=embeddings.using(model=settings.TEXT_EMBEDDING_MODEL),
                if_exists="replace_force",
            )

            logger.info(f"Successfully created Audio Index for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Audio Index: {e}")
            return False

    def create_description_index(self, video_id: str) -> bool:
        """Create Description Index with GPT-4o-mini descriptions.

        Args:
            video_id: Video identifier

        Returns:
            True if successful
        """
        # Ensure Pixeltable is initialized before any operations
        try:
            from ...api import _ensure_pixeltable
            _ensure_pixeltable()
        except ImportError:
            # If we're not in the API context, Pixeltable should already be initialized
            pass

        try:
            logger.info(f"Creating Description Index for video {video_id}")

            # Get video info from registry
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            # Check if frames view exists (created by create_image_index)
            try:
                frames_view = pxt.get_table(video_info.frames_view_name)
            except Exception as e:
                logger.error(f"Frames view not found: {e}. Make sure Image Index is created first.")
                return False

            # Step 9: use Pixeltable's native async vision UDF instead of our custom
            # synchronous one. Pixeltable runs these concurrently with adaptive
            # rate-limit throttling from OpenAI's response headers, which is the
            # main performance lever for per-frame description. Prompt is a literal
            # string; the image comes from the view column.
            logger.info("Adding description column via pxt_openai.vision (async + throttled)")
            frames_view.add_computed_column(
                description=pxt_openai.vision(
                    "Describe what is happening in this image in detail. "
                    "Be specific about objects, people, actions, and setting.",
                    frames_view.resized_frame,
                    model=settings.IMAGE_CAPTION_MODEL,
                    model_kwargs={
                        "max_tokens": settings.VISION_MAX_TOKENS,
                        "temperature": settings.VISION_TEMPERATURE,
                    },
                ),
                if_exists="ignore",
            )

            # Create text embedding index for descriptions
            logger.info("Creating text embedding index for descriptions")
            frames_view.add_embedding_index(
                column=frames_view.description,
                string_embed=embeddings.using(model=settings.TEXT_EMBEDDING_MODEL),
                if_exists="replace_force",
            )

            logger.info(f"Successfully created Description Index for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Description Index: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def create_domain_index(
        self, video_id: str, domain_context: str
    ) -> Optional[str]:
        """Create a Pixeltable domain view for a (video, domain_context) pair.

        Post-Step-8 architecture:
            * View name is ``{video_table_name}_domain_{context_hash}``,
              where ``context_hash`` is an 8-char blake2b of the
              lower/stripped domain context. One context → one view.
            * The view derives from the frames view and adds a
              ``domain_caption`` computed column driven by
              ``pxt_openai.vision`` (Pixeltable-native async UDF with
              adaptive throttling) with the domain context baked into the
              literal prompt.
            * A text-embedding index is built on ``domain_caption`` so search
              is semantic (``.similarity()``), mirroring Description Index.

        Returns the view name on success, ``None`` on failure. Callers
        (both the eager upload path and the lazy ``/chat`` path) use the
        return to know whether to surface an error.
        """
        # Ensure Pixeltable is initialized before any operations
        try:
            from ...api import _ensure_pixeltable
            _ensure_pixeltable()
        except ImportError:
            # If we're not in the API context, Pixeltable should already be initialized
            pass

        with monitor_processing(f"Domain index creation for {video_id}"):
            try:
                if not domain_context or not isinstance(domain_context, str):
                    raise ValueError("domain_context must be a non-empty string")

                context_hash = hash_domain_context(domain_context)
                logger.info(
                    f"Creating Domain Index for video {video_id} "
                    f"(context='{domain_context}', hash={context_hash})"
                )

                video_info = get_video_from_registry(video_id)
                if not video_info:
                    raise ValueError(f"Video {video_id} not found in registry")

                # Domain Index depends on the frames view (built by create_image_index).
                try:
                    frames_view = video_info.frames_view
                    if frames_view.count() == 0:
                        logger.error(f"Frames view for {video_id} is empty; cannot build Domain Index")
                        return None
                except Exception as e:
                    logger.error(
                        f"Frames view not accessible for {video_id}; "
                        f"Domain Index requires Image Index to have succeeded. Error: {e}"
                    )
                    return None

                domain_view_name = build_domain_view_name(
                    video_info.video_table_name, context_hash
                )

                logger.info(f"Creating domain view: {domain_view_name}")
                domain_view = pxt.create_view(
                    domain_view_name,
                    frames_view,
                    if_exists="replace_force",
                )

                # Step 9: Pixeltable-native vision UDF with the domain context baked
                # into the prompt literal. Same async+throttled path as the
                # description index — no per-row context column needed.
                domain_prompt = (
                    f"Analyze this image in the context of: {domain_context}\n\n"
                    f"Describe what you see with specific focus on elements relevant to "
                    f"{domain_context}. Be detailed about objects, actions, and visual "
                    f"elements that would be important in this domain context."
                )
                logger.info("Adding domain_caption computed column via pxt_openai.vision")
                domain_view.add_computed_column(
                    domain_caption=pxt_openai.vision(
                        domain_prompt,
                        domain_view.resized_frame,
                        model=settings.IMAGE_CAPTION_MODEL,
                        model_kwargs={
                            "max_tokens": settings.VISION_MAX_TOKENS,
                            "temperature": settings.VISION_TEMPERATURE,
                        },
                    ),
                    if_exists="replace_force",
                )

                # Mirror the audio-index fix from Step 6: materialize captions before
                # building the embedding index so failures are loud and catchable.
                logger.info("Pre-computing domain captions")
                domain_view.select(domain_view.domain_caption).collect()

                logger.info("Creating text embedding index on domain_caption")
                domain_view.add_embedding_index(
                    column=domain_view.domain_caption,
                    string_embed=embeddings.using(model=settings.TEXT_EMBEDDING_MODEL),
                    if_exists="replace_force",
                )

                add_domain_view(video_id, domain_context, domain_view_name)
                logger.info(
                    f"Successfully created Domain Index for video {video_id} "
                    f"(context='{domain_context}', view={domain_view_name})"
                )
                return domain_view_name

            except Exception as e:
                logger.error(f"Failed to create Domain Index: {e}")
                import traceback
                logger.debug(f"Domain index traceback: {traceback.format_exc()}")
                return None


# Global indexer instance
_indexer = None


def get_indexer() -> VideoIndexer:
    """Get or create global indexer instance.

    Returns:
        VideoIndexer instance
    """
    global _indexer
    if _indexer is None:
        _indexer = VideoIndexer()
    return _indexer


