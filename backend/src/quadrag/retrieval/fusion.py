"""Result fusion logic for combining multi-index retrieval results."""

from typing import Dict, List, Optional

from loguru import logger

from quadrag.config import get_settings
from quadrag.models import IndexType, RetrievalResult

settings = get_settings()
logger = logger.bind(name="ResultFusion")


class ResultFusion:
    """Fuses retrieval results from multiple indexes."""

    def __init__(self):
        """Initialize the result fusion."""
        self.weights = {
            IndexType.AUDIO: settings.WEIGHT_AUDIO,
            IndexType.IMAGE: settings.WEIGHT_IMAGE,
            IndexType.DESCRIPTION: settings.WEIGHT_DESCRIPTION,
            IndexType.DOMAIN: settings.WEIGHT_DOMAIN,
        }
        logger.info(f"ResultFusion initialized with weights: {self.weights}")

    def normalize_scores(
        self, results: Dict[IndexType, List[RetrievalResult]]
    ) -> Dict[IndexType, List[RetrievalResult]]:
        """Normalize similarity scores across all indexes.

        Args:
            results: Dictionary of index type to results

        Returns:
            Dictionary with normalized scores
        """
        # Find min and max scores across all results
        all_scores = []
        for result_list in results.values():
            all_scores.extend([r.similarity for r in result_list])

        if not all_scores:
            return results

        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range == 0:
            return results

        # Normalize scores to [0, 1]
        normalized_results = {}
        for index_type, result_list in results.items():
            normalized_list = []
            for result in result_list:
                normalized_similarity = (result.similarity - min_score) / score_range
                normalized_result = RetrievalResult(
                    content=result.content,
                    timestamp=result.timestamp,
                    similarity=normalized_similarity,
                    source=result.source,
                )
                normalized_list.append(normalized_result)
            normalized_results[index_type] = normalized_list

        return normalized_results

    def apply_weights(
        self, results: Dict[IndexType, List[RetrievalResult]]
    ) -> List[RetrievalResult]:
        """Apply index-specific weights to results.

        Args:
            results: Dictionary of index type to results

        Returns:
            Flat list of weighted results
        """
        weighted_results = []

        for index_type, result_list in results.items():
            weight = self.weights.get(index_type, 1.0)
            for result in result_list:
                weighted_result = RetrievalResult(
                    content=result.content,
                    timestamp=result.timestamp,
                    similarity=result.similarity * weight,
                    source=result.source,
                )
                weighted_results.append(weighted_result)

        return weighted_results

    def deduplicate_by_timestamp(
        self,
        results: List[RetrievalResult],
        time_window: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Deduplicate results with similar timestamps, keeping highest score.

        Args:
            results: List of retrieval results
            time_window: Time window in seconds for deduplication; defaults to
                ``settings.FUSION_DEDUP_WINDOW_SEC`` when ``None``.

        Returns:
            Deduplicated list of results
        """
        if time_window is None:
            time_window = settings.FUSION_DEDUP_WINDOW_SEC
        if not results:
            return []

        # Sort by similarity descending
        sorted_results = sorted(results, key=lambda r: r.similarity, reverse=True)

        deduplicated = []
        used_timestamps = []

        for result in sorted_results:
            # Check if this timestamp is too close to any already used
            is_duplicate = False
            for used_ts in used_timestamps:
                if abs(result.timestamp - used_ts) < time_window:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(result)
                used_timestamps.append(result.timestamp)

        return deduplicated

    def fuse_results(
        self,
        results: Dict[IndexType, List[RetrievalResult]],
        top_k: int = None,
        deduplicate: bool = True,
    ) -> List[RetrievalResult]:
        """Fuse results from multiple indexes.

        Args:
            results: Dictionary of index type to results
            top_k: Number of top results to return
            deduplicate: Whether to deduplicate by timestamp

        Returns:
            Fused and ranked list of results
        """
        if top_k is None:
            top_k = settings.FUSION_TOP_K

        logger.info(f"Fusing results from {len(results)} indexes")

        # Step 1: Normalize scores
        normalized_results = self.normalize_scores(results)

        # Step 2: Apply weights
        weighted_results = self.apply_weights(normalized_results)

        # Step 3: Deduplicate if requested
        if deduplicate:
            weighted_results = self.deduplicate_by_timestamp(weighted_results)

        # Step 4: Sort by weighted similarity
        fused_results = sorted(
            weighted_results, key=lambda r: r.similarity, reverse=True
        )

        # Step 5: Take top-k
        final_results = fused_results[:top_k]

        logger.info(
            f"Fused {len(weighted_results)} results, returning top {len(final_results)}"
        )

        return final_results

    def group_by_source(
        self, results: List[RetrievalResult]
    ) -> Dict[IndexType, List[RetrievalResult]]:
        """Group results by their source index.

        Args:
            results: List of retrieval results

        Returns:
            Dictionary mapping index type to results
        """
        grouped = {
            IndexType.IMAGE: [],
            IndexType.AUDIO: [],
            IndexType.DESCRIPTION: [],
            IndexType.DOMAIN: [],
        }

        for result in results:
            grouped[result.source].append(result)

        return grouped


# Global fusion instance
_fusion = None


def get_fusion() -> ResultFusion:
    """Get or create global fusion instance.

    Returns:
        ResultFusion instance
    """
    global _fusion
    if _fusion is None:
        _fusion = ResultFusion()
    return _fusion


