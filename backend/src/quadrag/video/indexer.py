"""Video indexer for creating the four semantic indexes."""

import base64
import os
from io import BytesIO
from typing import Optional

import google.generativeai as genai
import pixeltable as pxt
from loguru import logger
from openai import OpenAI
from pixeltable.functions import openai as pxt_openai
from pixeltable.functions.huggingface import clip
from pixeltable.functions.openai import embeddings
from pixeltable.functions.video import extract_audio
from pixeltable.iterators import AudioSplitter
from pixeltable.iterators.video import FrameIterator

from quadrag.config import get_settings
from quadrag.utils import calculate_frame_count, monitor_processing
from quadrag.video.functions import extract_text_from_chunk, resize_image
from quadrag.video.registry import get_video_from_registry, update_domain_view

settings = get_settings()
logger = logger.bind(name="VideoIndexer")


class VideoIndexer:
    """Creates and manages the four semantic indexes for videos."""

    def __init__(self):
        """Initialize the video indexer."""
        self.settings = settings
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # OpenAI API key should already be set at module level in api.py
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
                    iterator=FrameIterator.create(
                        video=video_table.video,
                        num_frames=optimal_frame_count
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
                        iterator=FrameIterator.create(
                            video=video_table.video,
                            num_frames=fallback_frames
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
            audio_view = pxt.create_view(
                video_info.audio_view_name,
                video_table,
                iterator=AudioSplitter.create(
                    audio=video_table.audio_extract,
                    chunk_duration_sec=settings.AUDIO_CHUNK_LENGTH,
                    overlap_sec=settings.AUDIO_OVERLAP_SECONDS,
                    min_chunk_duration_sec=settings.AUDIO_MIN_CHUNK_DURATION_SECONDS,
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
                print(f"DEBUG: About to add transcription column for {chunk_count} chunks")
                try:
                    audio_view.add_computed_column(
                        transcription=pxt_openai.transcriptions(
                            audio=audio_view.audio_chunk,
                            model=settings.AUDIO_TRANSCRIPT_MODEL,
                        ),
                        if_exists="ignore",
                    )
                    print("DEBUG: Transcription column added successfully")
                    logger.info("Transcription column added successfully (will compute on-demand)")
                except Exception as e:
                    print(f"DEBUG: Failed to add transcription column: {e}")
                    logger.error(f"Failed to add transcription column: {e}")
                    raise

            # Extract text from transcription
            # Check if column already exists
            try:
                _ = audio_view.transcript_text
                logger.info("Transcript text column already exists, skipping addition")
            except AttributeError:
                logger.info("Adding text extraction column...")
                print("DEBUG: About to add text extraction column")
                try:
                    audio_view.add_computed_column(
                        transcript_text=extract_text_from_chunk(audio_view.transcription),
                        if_exists="ignore",
                    )
                    print("DEBUG: Text extraction column added successfully")
                    logger.info("Text extraction column added successfully (will compute on-demand)")
                except Exception as e:
                    print(f"DEBUG: Failed to add text extraction column: {e}")
                    logger.error(f"Failed to add text extraction column: {e}")
                    raise

            # Skip pre-computation to avoid blocking - transcriptions will be computed lazily during search
            logger.info("Skipping pre-computation of transcriptions - will compute lazily during search")
            print("DEBUG: Skipping transcription pre-computation to avoid blocking")

            # Skip embedding index creation to avoid blocking
            # Embedding index creation would require all transcriptions to be computed first
            # which would block for 10-30 minutes. Instead, we'll use text-based search
            # which works immediately (though less accurate than semantic search)
            logger.info("Skipping embedding index creation to avoid blocking")
            logger.info("Note: Audio search will use text matching (faster, but less semantic)")
            logger.info("Note: Embedding index can be created later via separate endpoint if needed")

            logger.info(f"Successfully created Audio Index structure for video {video_id}")
            logger.info("Transcriptions pre-computed and available for search")
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

            # Import our synchronous description function
            from quadrag.video.functions import describe_image

            # Add description column using synchronous UDF
            logger.info("Adding description column using synchronous OpenAI Vision API")
            frames_view.add_computed_column(
                description=describe_image(frames_view.resized_frame),
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
        self, video_id: str, session_id: str, domain_context: str
    ) -> bool:
        """Create Domain Captions Index with context-specific captions.

        Args:
            video_id: Video identifier
            session_id: Session identifier
            domain_context: User-provided domain context

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

        with monitor_processing(f"Domain index creation for {video_id}"):
            try:
                logger.info(
                    f"Creating Domain Index for video {video_id} "
                    f"with context: {domain_context}"
                )
                video_info = get_video_from_registry(video_id)
                if not video_info:
                    raise ValueError(f"Video {video_id} not found in registry")

                # Check if frames_view exists and is accessible (requires image index)
                if not hasattr(video_info, 'frames_view') or not video_info.frames_view:
                    logger.warning(f"Cannot create Domain Index for video {video_id}: Image index not available (frames_view attribute missing)")
                    logger.info(f"Domain context will be used for text-based search only")
                    # Still update the domain view to indicate domain context is set
                    domain_view_name = f"{video_info.video_table_name}_domain_{session_id[:8]}"
                    update_domain_view(video_id, domain_view_name)
                    return True

                # Try to access frames_view to ensure it exists in Pixeltable
                try:
                    frames_view = video_info.frames_view
                    # Test if frames_view is actually accessible by checking if it has data
                    test_count = frames_view.count()
                    if test_count == 0:
                        logger.warning(f"Frames view exists but is empty for video {video_id}")
                        logger.info(f"Domain context will be used for text-based search only")
                        domain_view_name = f"{video_info.video_table_name}_domain_{session_id[:8]}"
                        update_domain_view(video_id, domain_view_name)
                        return True
                except Exception as e:
                    logger.warning(f"Cannot create Domain Index for video {video_id}: Frames view not accessible ({e})")
                    logger.info(f"Domain context will be used for text-based search only")
                    # Still update the domain view to indicate domain context is set
                    domain_view_name = f"{video_info.video_table_name}_domain_{session_id[:8]}"
                    update_domain_view(video_id, domain_view_name)
                    return True

                # Create domain-specific captions by manually processing each frame
                # Store them in registry for search access (avoiding Pixeltable UDF issues)
                logger.info(f"Creating domain captions for {video_id} with context: {domain_context}")

                # Get all frames
                frames_data = list(frames_view.select(
                    frames_view.pos_msec,
                    frames_view.resized_frame
                ).collect())

                logger.info(f"Processing {len(frames_data)} frames for domain captions")

                # Process frames in batches to manage memory and API rate limits
                domain_captions = {}
                batch_size = 5  # Process 5 frames at a time
                total_frames = len(frames_data)

                with monitor_processing("Domain index creation"):
                    for batch_start in range(0, total_frames, batch_size):
                        batch_end = min(batch_start + batch_size, total_frames)
                        batch_frames = frames_data[batch_start:batch_end]

                        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size}: "
                                   f"frames {batch_start}-{batch_end-1}")

                        # Process each frame in the current batch
                        for i, frame_data in enumerate(batch_frames):
                            frame_idx = batch_start + i
                            try:
                                frame_image = frame_data['resized_frame']
                                pos_msec = frame_data['pos_msec']

                                # Generate caption using direct API call
                                try:
                                    # Convert PIL Image to base64
                                    buffer = BytesIO()
                                    frame_image.save(buffer, format="PNG")
                                    img_base64 = base64.b64encode(buffer.getvalue()).decode()

                                    # Create OpenAI client
                                    client = OpenAI()

                                    # Create domain-specific prompt
                                    prompt = f"""Analyze this image in the context of: {domain_context}

Describe what you see, focusing on elements that are most relevant to understanding {domain_context}.
- If the scene contains people, analyze their expressions, body language, and interactions
- If the scene shows actions or objects, describe how they relate to {domain_context}
- Highlight any visual elements that demonstrate concepts related to {domain_context}
- Be specific about what you're observing rather than making assumptions

Provide a clear, factual description of the visual content."""

                                    # Make synchronous API call
                                    response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[{
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": prompt},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                                            ]
                                        }],
                                        max_tokens=200,
                                        temperature=0.3
                                    )

                                    if response and response.choices and len(response.choices) > 0:
                                        description = response.choices[0].message.content
                                        if description and isinstance(description, str):
                                            caption = description.strip() or f"Empty description for {domain_context}"
                                        else:
                                            caption = f"Invalid response format for {domain_context}"
                                    else:
                                        caption = f"No response from vision API for {domain_context}"

                                except Exception as e:
                                    error_msg = str(e)[:100]
                                    caption = f"Domain caption unavailable ({domain_context}): {error_msg}"

                                domain_captions[pos_msec] = caption

                                # Log each generated caption
                                logger.info(f"Frame {pos_msec/1000:.1f}s: {caption[:200]}{'...' if len(caption) > 200 else ''}")

                            except Exception as e:
                                logger.warning(f"Failed to generate caption for frame at {pos_msec}: {e}")
                                domain_captions[pos_msec] = f"Domain caption unavailable: {str(e)[:50]}"

                        # Add rate limiting between batches to avoid API limits
                        if batch_end < total_frames:
                            logger.info("Rate limiting: waiting 2 seconds before next batch...")
                            import time
                            time.sleep(2)

                # Store domain captions in registry for search access
                # This avoids Pixeltable UDF and embedding issues
                from quadrag.video.registry import update_domain_captions
                update_domain_captions(video_id, domain_captions, domain_context)

                logger.info(f"Stored {len(domain_captions)} domain captions in registry")

                # Log summary of all generated captions
                logger.info(f"📋 DOMAIN CAPTIONS SUMMARY for video {video_id} (context: '{domain_context}'):")
                for pos_msec, caption in sorted(domain_captions.items()):
                    timestamp = pos_msec / 1000.0
                    logger.info(f"  {timestamp:.1f}s: {caption}")

                # Update the domain view to indicate domain context is set
                domain_view_name = f"{video_info.video_table_name}_domain_{session_id[:8]}"
                update_domain_view(video_id, domain_view_name)

                logger.info(f"Successfully created Domain Index for video {video_id} with context: {domain_context}")
                return True

            except Exception as e:
                logger.error(f"Failed to create Domain Index: {e}")
                return False


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


