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
            audio_view = self.video_info.audio_view

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
                
                # Get limited chunks and force computation of transcriptions
                # This ensures some transcriptions are computed for search
                limited_chunks = []
                try:
                    import asyncio

                    # Use asyncio.run_in_executor to avoid event loop conflicts
                    def collect_chunks():
                        try:
                            # First try to get chunks with transcriptions computed
                            chunks = audio_view.select(
                                audio_view.start_time_sec,
                                audio_view.end_time_sec,
                                audio_view.transcript_text,
                            ).limit(10).collect()  # Start with fewer chunks to force computation
                            return chunks
                        except Exception as e:
                            logger.warning(f"Failed to collect audio chunks: {e}")
                            return []

                    # Run in thread pool to avoid event loop issues
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        limited_chunks = loop.run_until_complete(
                            asyncio.get_event_loop().run_in_executor(None, collect_chunks)
                        )
                    finally:
                        loop.close()

                    # If we got chunks but they have empty transcriptions, try to trigger computation
                    if limited_chunks and any(not str(chunk.get("transcript_text", "")).strip() for chunk in limited_chunks):
                        logger.info("Triggering transcription computation for search...")
                        # Force computation by accessing the transcript_text field
                        for chunk in limited_chunks[:5]:  # Compute first 5
                            try:
                                _ = str(chunk.get("transcript_text", ""))
                            except Exception as e:
                                logger.debug(f"Failed to compute transcription for chunk: {e}")

                        # Re-fetch to get computed transcriptions
                        def collect_chunks_again():
                            try:
                                chunks = audio_view.select(
                                    audio_view.start_time_sec,
                                    audio_view.end_time_sec,
                                    audio_view.transcript_text,
                                ).limit(10).collect()
                                return chunks
                            except Exception as e:
                                logger.warning(f"Failed to re-collect audio chunks: {e}")
                                return limited_chunks  # Return original chunks if re-collection fails

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            limited_chunks = loop.run_until_complete(
                                asyncio.get_event_loop().run_in_executor(None, collect_chunks_again)
                            )
                        finally:
                            loop.close()

                except Exception as e:
                    logger.warning(f"Failed to collect audio chunks with executor: {e}")
                    limited_chunks = []
                
                # Simple text matching (case-insensitive)
                query_lower = query_text.lower()
                scored_chunks = []
                for chunk in limited_chunks:
                    text = str(chunk.get("transcript_text", "")).lower().strip()
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
                        logger.debug(f"Match found in chunk {chunk.get('start_time_sec', 0)}: '{text[:100]}...' (score: {score})")
                
                # Sort by score and take top_k
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                retrieval_results = [
                    RetrievalResult(
                        content=str(chunk["transcript_text"]),
                        timestamp=float(chunk["start_time_sec"]),
                        similarity=float(score),
                        source=IndexType.AUDIO,
                    )
                    for chunk, score in scored_chunks[:top_k]
                ]
                
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
            if top_k is None:
                top_k = settings.TOP_K_DESCRIPTION

            logger.info(f"Searching Description Index with query: '{query_text[:50]}...'")

            # Check if frames_view exists
            if not hasattr(self.video_info, 'frames_view') or not self.video_info.frames_view:
                logger.info("Description Index not available (requires image index)")
                return []

            # Additional check: ensure frames_view actually exists in Pixeltable
            try:
                frames_view = self.video_info.frames_view
                # Try to access a property to verify the view exists
                _ = frames_view._name
            except Exception as e:
                logger.warning(f"Description Index not accessible: {e}")
                return []

            frames_view = self.video_info.frames_view

            # Perform similarity search on descriptions
            sims = frames_view.description.similarity(query_text)
            results = frames_view.select(
                frames_view.pos_msec,
                frames_view.description,
                similarity=sims,
            ).order_by(sims, asc=False).limit(top_k)

            # Convert to RetrievalResult
            retrieval_results = []
            for entry in results.collect():
                retrieval_results.append(
                    RetrievalResult(
                        content=entry["description"],
                        timestamp=float(entry["pos_msec"]) / 1000.0,
                        similarity=float(entry["similarity"]),
                        source=IndexType.DESCRIPTION,
                    )
                )

            logger.info(f"Found {len(retrieval_results)} description results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error searching Description Index: {e}")
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
            if not self.session_id:
                logger.warning("No session_id provided for domain search")
                return []

            if top_k is None:
                top_k = settings.TOP_K_DOMAIN

            logger.info(f"Searching Domain Index with query: '{query_text[:50]}...'")

            # Check if frames_view exists
            if not hasattr(self.video_info, 'frames_view') or not self.video_info.frames_view:
                logger.info("Domain Index not available (requires image index)")
                return []

            # Additional check: ensure frames_view actually exists in Pixeltable
            try:
                frames_view = self.video_info.frames_view
                # Try to access a property to verify the view exists
                _ = frames_view._name
            except Exception as e:
                logger.warning(f"Domain Index not accessible: {e}")
                return []
            
            # Get the domain caption column for this session
            column_name = f"domain_caption_{self.session_id[:8]}"
            
            if not hasattr(frames_view, column_name):
                logger.warning(f"Domain caption column {column_name} not found")
                return []

            # Perform similarity search on domain captions
            domain_column = getattr(frames_view, column_name)
            sims = domain_column.similarity(query_text)
            results = frames_view.select(
                frames_view.pos_msec,
                domain_column,
                similarity=sims,
            ).order_by(sims, asc=False).limit(top_k)

            # Convert to RetrievalResult
            retrieval_results = []
            for entry in results.collect():
                retrieval_results.append(
                    RetrievalResult(
                        content=entry[column_name],
                        timestamp=float(entry["pos_msec"]) / 1000.0,
                        similarity=float(entry["similarity"]),
                        source=IndexType.DOMAIN,
                    )
                )

            logger.info(f"Found {len(retrieval_results)} domain results")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error searching Domain Index: {e}")
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


