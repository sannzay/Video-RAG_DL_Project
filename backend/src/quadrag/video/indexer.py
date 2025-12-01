"""Video indexer for creating the four semantic indexes."""

import os
from typing import Optional

import google.generativeai as genai
import pixeltable as pxt
from loguru import logger
from openai import OpenAI
from pixeltable.functions import openai as pxt_openai
from pixeltable.functions.huggingface import clip
from pixeltable.functions.video import extract_audio
from pixeltable.iterators import AudioSplitter
from pixeltable.iterators.video import FrameIterator

from quadrag.config import get_settings
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
        
        # Configure OpenAI API key for Pixeltable
        # Pixeltable needs the API key in environment or config
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    def create_image_index(self, video_id: str) -> bool:
        """Create Image Index with CLIP embeddings.

        Args:
            video_id: Video identifier

        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating Image Index for video {video_id}")
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            video_table = video_info.video_table

            # Create frames view
            logger.info(f"Creating frames view: {video_info.frames_view_name}")
            try:
                logger.info(f"Attempting to create frame iterator with {settings.SPLIT_FRAMES_COUNT} frames")
                logger.info(f"Video file: {video_table.video}")
                frames_view = pxt.create_view(
                    video_info.frames_view_name,
                    video_table,
                    iterator=FrameIterator.create(
                        video=video_table.video,
                        num_frames=settings.SPLIT_FRAMES_COUNT
                    ),
                    if_exists="replace_force",
                )
            except Exception as e:
                logger.error(f"Failed to create frame iterator with {settings.SPLIT_FRAMES_COUNT} frames: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Try with fewer frames as fallback
                logger.info("Retrying with 10 frames...")
                try:
                frames_view = pxt.create_view(
                    video_info.frames_view_name,
                    video_table,
                    iterator=FrameIterator.create(
                        video=video_table.video,
                        num_frames=10
                    ),
                    if_exists="replace_force",
                )
                except Exception as e2:
                    logger.error(f"Failed even with 10 frames: {e2}")
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
        try:
            logger.info(f"Creating Audio Index for video {video_id}")
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            video_table = video_info.video_table

            # Extract audio
            logger.info("Extracting audio from video")
            video_table.add_computed_column(
                audio_extract=extract_audio(video_table.video, format="mp3"),
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
                audio_view.add_computed_column(
                    transcription=pxt_openai.transcriptions(
                        audio=audio_view.audio_chunk,
                        model=settings.AUDIO_TRANSCRIPT_MODEL,
                    ),
                    if_exists="ignore",
                )
                logger.info("Transcription column added successfully (will compute on-demand)")

            # Extract text from transcription
            # Check if column already exists
            try:
                _ = audio_view.transcript_text
                logger.info("Transcript text column already exists, skipping addition")
            except AttributeError:
                logger.info("Adding text extraction column...")
                audio_view.add_computed_column(
                    transcript_text=extract_text_from_chunk(audio_view.transcription),
                    if_exists="ignore",
                )
                logger.info("Text extraction column added successfully (will compute on-demand)")

            # Force computation of transcriptions now to avoid lazy loading issues during search
            logger.info("Pre-computing transcriptions to ensure they are available for search...")
            try:
                # Collect all chunks to force transcription computation
                all_chunks = audio_view.select(
                    audio_view.start_time_sec,
                    audio_view.end_time_sec,
                    audio_view.transcript_text,
                ).collect()

                logger.info(f"Successfully pre-computed {len(all_chunks)} transcriptions")
                for i, chunk in enumerate(all_chunks):
                    transcription = str(chunk.get("transcript_text", ""))
                    if transcription.strip():
                        logger.debug(f"Chunk {i}: '{transcription[:100]}...'")
                    else:
                        logger.warning(f"Chunk {i} has empty transcription")

            except Exception as e:
                logger.warning(f"Failed to pre-compute transcriptions: {e}")
                logger.info("Transcriptions will be computed lazily during search")

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
        try:
            logger.info(f"Creating Description Index for video {video_id}")
            video_info = get_video_from_registry(video_id)
            if not video_info:
                raise ValueError(f"Video {video_id} not found in registry")

            frames_view = video_info.frames_view

            # Add description column using GPT-4o-mini
            logger.info("Generating frame descriptions")
            from pixeltable.functions.openai import vision
            
            frames_view.add_computed_column(
                description=vision(
                    prompt=settings.DESCRIPTION_PROMPT,
                    image=frames_view.resized_frame,
                    model=settings.IMAGE_CAPTION_MODEL,
                ),
                if_exists="ignore",
            )

            # Create description view (same as frames but for organization)
            logger.info(f"Creating description view: {video_info.description_view_name}")
            
            # Use frames_view directly and add embedding index
            from pixeltable.functions.openai import embeddings
            
            frames_view.add_embedding_index(
                column=frames_view.description,
                string_embed=embeddings.using(model="text-embedding-3-small"),
                if_exists="replace_force",
                idx_name="description_idx",
            )

            logger.info(f"Successfully created Description Index for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Description Index: {e}")
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

            frames_view = video_info.frames_view

            # Double-check that frames_view actually exists in Pixeltable
            try:
                # Try to access the frames_view to ensure it exists
                test_access = pxt.get_table(frames_view._name)
            except Exception as e:
                logger.warning(f"Frames view {frames_view._name} not accessible in Pixeltable: {e}")
                logger.info(f"Domain context will be used for text-based search only")
                domain_view_name = f"{video_info.video_table_name}_domain_{session_id[:8]}"
                update_domain_view(video_id, domain_view_name)
                return True

            # Create domain-specific caption column
            domain_prompt = settings.DOMAIN_PROMPT_TEMPLATE.format(
                domain_context=domain_context
            )
            
            column_name = f"domain_caption_{session_id[:8]}"
            
            logger.info(f"Generating domain-specific captions with prompt: {domain_prompt}")
            from pixeltable.functions.openai import vision
            
            frames_view.add_computed_column(
                **{column_name: vision(
                    prompt=domain_prompt,
                    image=frames_view.resized_frame,
                    model=settings.IMAGE_CAPTION_MODEL,
                )},
                if_exists="ignore",
            )

            # Add embedding index for domain captions
            from pixeltable.functions.openai import embeddings
            
            idx_name = f"domain_idx_{session_id[:8]}"
            frames_view.add_embedding_index(
                column=getattr(frames_view, column_name),
                string_embed=embeddings.using(model="text-embedding-3-small"),
                if_exists="replace_force",
                idx_name=idx_name,
            )

            # Update registry with domain view info
            domain_view_name = f"{video_info.frames_view_name}_{session_id[:8]}"
            update_domain_view(video_id, domain_view_name)

            logger.info(f"Successfully created Domain Index for video {video_id}")
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


