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
                # If embedding index doesn't exist, use text-based search across ALL chunks
                logger.warning(f"Embedding index not available, using text search across all chunks: {e}")

                # Get all audio chunks data - now running in proper async context
                try:
                    logger.info("Collecting all audio chunks with pre-computed transcriptions...")

                    # Collect ALL chunks, not just a limited subset
                    all_chunks = audio_view.select(
                        audio_view.start_time_sec,
                        audio_view.end_time_sec,
                        audio_view.transcript_text,
                    ).collect()

                    logger.info(f"Successfully collected {len(all_chunks)} audio chunks for comprehensive search")

                    # Log transcription status for debugging (only first few for brevity)
                    sample_size = min(5, len(all_chunks))
                    for i in range(sample_size):
                        chunk = all_chunks[i]
                        transcription = str(chunk.get("transcript_text", ""))
                        raw_transcript = chunk.get("transcription", "")
                        logger.debug(f"Sample chunk {i}: transcript_text='{transcription[:100]}...', raw_transcription='{str(raw_transcript)[:200]}...'")
                        if not transcription.strip():
                            logger.warning(f"Chunk {i} has empty transcription")

                    if len(all_chunks) > sample_size:
                        logger.debug(f"... and {len(all_chunks) - sample_size} more chunks")

                except Exception as e:
                    logger.warning(f"Failed to collect audio chunks: {e}")
                    all_chunks = []

                # Simple text matching across ALL chunks (case-insensitive)
                query_lower = query_text.lower()
                scored_chunks = []
                for chunk in all_chunks:
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
            if top_k is None:
                top_k = settings.TOP_K_DESCRIPTION

            logger.info(f"Searching Description Index with query: '{query_text[:50]}...'")

            # Check if frames view exists
            if not self.video_info.frames_view_name:
                logger.info("Description Index not available (no frames view)")
                return []

            # Try to get the frames view safely
            try:
                frames_view = self.video_info.frames_view
            except Exception as e:
                logger.error(f"Description Index not accessible: {e}")
                return []

            # Try similarity search on description column
            try:
                sims = frames_view.description.similarity(query_text)
                results = frames_view.select(
                    frames_view.pos_msec,
                    frames_view.description,
                    similarity=sims,
                ).order_by(sims, asc=False).limit(top_k)

                # Convert to RetrievalResult
                retrieval_results = []
                try:
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
                    logger.error(f"Failed to collect description results: {e}")
                    return []

            except Exception as e:
                logger.error(f"Description similarity search failed: {e}")
                logger.info("Description column may not exist or embedding index not available")
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
            if not self.session_id:
                logger.warning("No session_id provided for domain search")
                return []

            if top_k is None:
                top_k = settings.TOP_K_DOMAIN

            logger.info(f"Searching Domain Index with query: '{query_text[:50]}...'")

            # Check if frames_view exists (avoid triggering property getter that calls pxt.get_table)
            if not self.video_info.frames_view_name:
                logger.info("Domain Index not available (requires image index)")
                return []

            # Try to get the frames view safely
            try:
                frames_view = self.video_info.frames_view
            except Exception as e:
                logger.warning(f"Domain Index not accessible: {e}")
                return []

            # Check if domain captions exist in registry
            if not hasattr(self.video_info, 'domain_captions') or not self.video_info.domain_captions:
                logger.warning("Domain captions not found in registry")
                return []

            # Get domain captions from registry
            domain_captions = self.video_info.domain_captions
            domain_context = getattr(self.video_info, 'domain_context', 'unknown')
            logger.info(f"Found {len(domain_captions)} domain captions in registry (context: '{domain_context}')")

            # Since we can't use Pixeltable embeddings for stored data,
            # we'll do a simple text similarity search
            import difflib

            # Score each caption against the query
            scored_captions = []
            query_lower = query_text.lower()

            for pos_msec, caption in domain_captions.items():
                caption_lower = caption.lower()

                # Primary: sequence matching for overall similarity
                sequence_similarity = difflib.SequenceMatcher(None, query_lower, caption_lower).ratio()

                # Secondary: keyword matching - check if query words appear in caption
                query_words = set(query_lower.split())
                caption_words = set(caption_lower.split())

                # Calculate word overlap
                if query_words:
                    word_overlap = len(query_words.intersection(caption_words)) / len(query_words)
                else:
                    word_overlap = 0.0

                # Combined similarity: weighted average of sequence and word matching
                combined_similarity = (sequence_similarity * 0.7) + (word_overlap * 0.3)

                scored_captions.append({
                    'pos_msec': pos_msec,
                    'caption': caption,
                    'similarity': combined_similarity,
                    'sequence_sim': sequence_similarity,
                    'word_overlap': word_overlap
                })

            # Sort by similarity and take top_k
            scored_captions.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = scored_captions[:top_k]

            # Log search results
            logger.info(f"🔍 DOMAIN SEARCH RESULTS for '{query_text[:50]}...':")
            for i, item in enumerate(top_results[:3]):  # Show top 3 results
                similarity_pct = item['similarity'] * 100
                sequence_pct = item.get('sequence_sim', 0) * 100
                overlap_pct = item.get('word_overlap', 0) * 100
                timestamp = item['pos_msec'] / 1000.0
                logger.info(f"  #{i+1} ({similarity_pct:.1f}% seq:{sequence_pct:.1f}% word:{overlap_pct:.1f}%): {timestamp:.1f}s - {item['caption'][:100]}{'...' if len(item['caption']) > 100 else ''}")

            # Convert to RetrievalResult
            retrieval_results = []
            for item in top_results:
                # Lower threshold for domain captions since they're long and detailed
                # Accept any result with similarity > 0.01 (1%) as domain captions are comprehensive
                if item['similarity'] > 0.01:
                    retrieval_results.append(
                        RetrievalResult(
                            content=item['caption'],
                            timestamp=float(item['pos_msec']) / 1000.0,
                            similarity=float(item['similarity']),
                            source=IndexType.DOMAIN,
                        )
                    )

            logger.info(f"Found {len(retrieval_results)} domain results")
            return retrieval_results

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

        # For text queries, we don't search image index directly (use description index instead)
        # Image index is only used for image-to-image similarity when query_image is provided
        if query_image:
            results[IndexType.IMAGE] = self.search_image_index(query_image)
        else:
            results[IndexType.IMAGE] = []

        return results


