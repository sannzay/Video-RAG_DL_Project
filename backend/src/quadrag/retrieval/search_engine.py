"""Multi-index video search engine."""

from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

from quadrag.config import get_settings
from quadrag.models import IndexType, RetrievalResult
from quadrag.video.registry import get_video_from_registry

settings = get_settings()
logger = logger.bind(name="SearchEngine")


class VideoSearchEngine:
    """Search engine for querying all four semantic indexes."""

    def __init__(self, video_id: str, session_id: Optional[str] = None):
        """Initialize the search engine.

        Args:
            video_id: Video identifier
            session_id: Optional session ID for domain-specific search

        Raises:
            ValueError: If video not found in registry
        """
        self.video_id = video_id
        self.session_id = session_id
        self.video_info = get_video_from_registry(video_id)
        
        if not self.video_info:
            raise ValueError(f"Video {video_id} not found in registry")

        logger.info(f"Initialized search engine for video {video_id}")

    def search_image_index(
        self, query_image: Image.Image, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search Image Index using CLIP similarity.

        Args:
            query_image: PIL Image to search for
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        try:
            if top_k is None:
                top_k = settings.TOP_K_IMAGE

            logger.info(f"Searching Image Index with top_k={top_k}")
            frames_view = self.video_info.frames_view

            # Perform similarity search
            sims = frames_view.resized_frame.similarity(query_image)
            results = frames_view.select(
                frames_view.pos_msec,
                frames_view.resized_frame,
                similarity=sims,
            ).order_by(sims, asc=False).limit(top_k)

            # Convert to RetrievalResult
            retrieval_results = []
            for entry in results.collect():
                retrieval_results.append(
                    RetrievalResult(
                        content=f"Visual content at {entry['pos_msec']/1000:.2f}s",
                        timestamp=float(entry["pos_msec"]) / 1000.0,
                        similarity=float(entry["similarity"]),
                        source=IndexType.IMAGE,
                    )
                )

            logger.info(f"Found {len(retrieval_results)} image results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error searching Image Index: {e}")
            return []

    def search_audio_index(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search Audio Index using text similarity on transcripts.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        try:
            if top_k is None:
                top_k = settings.TOP_K_AUDIO

            logger.info(f"Searching Audio Index with query: '{query_text[:50]}...'")

            # Check if audio view exists
            if not self.video_info.audio_view_name:
                logger.info("Audio Index not available")
                return []

            # Try to get the audio view safely
            try:
                audio_view = self.video_info.audio_view
            except Exception as e:
                logger.error(f"Audio Index not accessible: {e}")
                return []

            # Try similarity search if embedding index exists
            try:
                sims = audio_view.transcript_text.similarity(query_text)
                results = audio_view.select(
                    audio_view.start_time_sec,
                    audio_view.end_time_sec,
                    audio_view.transcript_text,
                    similarity=sims,
                ).order_by(sims, asc=False).limit(top_k)
                
                # Convert to RetrievalResult - collect() can cause event loop issues
                retrieval_results = []
                try:
                    results_list = results.collect()
                    for entry in results_list:
                        retrieval_results.append(
                            RetrievalResult(
                                content=entry["transcript_text"],
                                timestamp=float(entry["start_time_sec"]),
                                similarity=float(entry["similarity"]),
                                source=IndexType.AUDIO,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to collect similarity results: {e}, falling back to text search")
                    retrieval_results = []
            except (AttributeError, Exception) as e:
                # If embedding index doesn't exist, use text-based search
                # Limit to first 50 chunks to avoid blocking on all transcriptions
                logger.warning(f"Embedding index not available, using text search on limited chunks: {e}")
                
                # Get all audio chunks data - now running in proper async context
                try:
                    logger.info("Collecting audio chunks with pre-computed transcriptions...")

                    # Since we're in async context, collect() should work
                    limited_chunks = audio_view.select(
                        audio_view.start_time_sec,
                        audio_view.end_time_sec,
                        audio_view.transcript_text,
                    ).limit(10).collect()

                    logger.info(f"Successfully collected {len(limited_chunks)} audio chunks")

                    # Log transcription status for debugging
                    for i, chunk in enumerate(limited_chunks):
                        transcription = str(chunk.get("transcript_text", ""))
                        raw_transcript = chunk.get("transcription", "")
                        logger.info(f"Chunk {i}: transcript_text='{transcription[:100]}...', raw_transcription='{str(raw_transcript)[:200]}...'")
                        if transcription.strip():
                            logger.debug(f"Chunk {i}: '{transcription[:100]}...'")
                        else:
                            logger.warning(f"Chunk {i} has empty transcription, raw: {raw_transcript}")

                except Exception as e:
                    logger.warning(f"Failed to collect audio chunks: {e}")
                    limited_chunks = []
                
                # Simple text matching (case-insensitive)
                query_lower = query_text.lower()
                scored_chunks = []
                for chunk in limited_chunks:
                    # Handle both dict and object access patterns
                    if isinstance(chunk, dict):
                        text = str(chunk.get("transcript_text", "")).lower().strip()
                        start_time = float(chunk.get("start_time_sec", 0))
                        transcript_text = str(chunk.get("transcript_text", ""))
                    else:
                        text = str(chunk.get("transcript_text", "")).lower().strip()
                        start_time = float(chunk.get("start_time_sec", 0))
                        transcript_text = str(chunk.get("transcript_text", ""))

                    if not text:
                        continue  # Skip empty transcriptions

                    # Check if query is contained in text or individual words match
                    contains_query = query_lower in text
                    word_matches = any(word in text for word in query_lower.split() if len(word) > 2)  # Only match words > 2 chars

                    if contains_query or word_matches:
                        # Simple score based on how many query words match
                        matching_words = sum(1 for word in query_lower.split() if word in text)
                        score = matching_words / max(len(query_lower.split()), 1)
                        scored_chunks.append((chunk, score))
                        logger.debug(f"Match found in chunk {start_time}: '{text[:100]}...' (score: {score})")

                # Sort by score and take top_k
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                retrieval_results = []
                for chunk, score in scored_chunks[:top_k]:
                    if isinstance(chunk, dict):
                        content = str(chunk.get("transcript_text", ""))
                        timestamp = float(chunk.get("start_time_sec", 0))
                    else:
                        content = str(chunk.get("transcript_text", ""))
                        timestamp = float(chunk.get("start_time_sec", 0))

                    retrieval_results.append(
                        RetrievalResult(
                            content=content,
                            timestamp=timestamp,
                            similarity=float(score),
                            source=IndexType.AUDIO,
                        )
                    )
                
                if len(retrieval_results) == 0:
                    logger.info("No matches found in first 50 chunks. More transcriptions may be needed.")

            logger.info(f"Found {len(retrieval_results)} audio results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error searching Audio Index: {e}")
            return []

    def search_description_index(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search Description Index using text similarity on frame descriptions.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        try:
            logger.info(f"Description Index search requested but disabled (no image index)")
            # Return empty results since description index is not created
            return []

        except Exception as e:
            logger.error(f"Error searching Description Index: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Description search traceback: {traceback.format_exc()}")
            return []

    def search_domain_index(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Search Domain Captions Index using text similarity.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        try:
            logger.info(f"Domain Index search requested but disabled (no image index)")
            logger.info(f"Domain context is available for text-based search enhancement")
            # Return empty results since domain index is not created
            return []

        except Exception as e:
            logger.error(f"Error searching Domain Index: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Domain search traceback: {traceback.format_exc()}")
            return []

    def search_all_indexes(
        self,
        query_text: str,
        query_image: Optional[Image.Image] = None,
        use_domain: bool = True,
    ) -> Dict[IndexType, List[RetrievalResult]]:
        """Search all relevant indexes.

        Args:
            query_text: Query text
            query_image: Optional query image
            use_domain: Whether to include domain index

        Returns:
            Dictionary mapping IndexType to list of results
        """
        results = {}

        # Search text-based indexes
        results[IndexType.AUDIO] = self.search_audio_index(query_text)
        results[IndexType.DESCRIPTION] = self.search_description_index(query_text)

        # Search domain index if available and requested
        if use_domain and self.session_id:
            results[IndexType.DOMAIN] = self.search_domain_index(query_text)
        else:
            results[IndexType.DOMAIN] = []

        # Search image index if image provided
        if query_image:
            results[IndexType.IMAGE] = self.search_image_index(query_image)
        else:
            results[IndexType.IMAGE] = []

        return results


