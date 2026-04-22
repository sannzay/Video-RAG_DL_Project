"""Lazy-create + LRU-evict wiring for multi-domain Pixeltable views.

Introduced in Step 8. The registry tracks one or more domain views per video
keyed by hash-of-normalized-context; this module is the glue that:

* Finds or creates a view for a given (video_id, domain_context) pair.
* Enforces the per-video cap from ``settings.MAX_DOMAIN_VIEWS_PER_VIDEO`` by
  dropping the least-recently-used view (both the Pixeltable handle and the
  registry entry) before creating a new one.

Kept out of ``indexer.py`` because ``VideoIndexer`` is an ML-heavy object
lazy-loaded from ``api.py``; this module depends only on the registry and
Pixeltable's drop primitive.
"""

from __future__ import annotations

from typing import Optional

import pixeltable as pxt
from loguru import logger

from quadrag.config import get_settings
from quadrag.video.registry import (
    drop_domain_view,
    get_video_from_registry,
    hash_domain_context,
    lru_domain_view,
    touch_domain_view,
)

_settings = get_settings()


def _evict_lru_if_needed(video_id: str, cap: int) -> None:
    """If the video already has ``cap`` views, drop the LRU one."""
    info = get_video_from_registry(video_id)
    if info is None:
        return
    while len(info.domain_views) >= cap:
        victim = lru_domain_view(video_id)
        if victim is None:
            return
        context_hash, view_name = victim
        # Drop the registry entry first so a concurrent caller can't hand the
        # about-to-be-deleted view out to a searcher.
        dropped_name = drop_domain_view(video_id, context_hash)
        if dropped_name is None:
            return
        try:
            pxt.drop_table(dropped_name, force=True)
            logger.info(
                f"LRU-evicted domain view for {video_id}: "
                f"hash={context_hash} view={dropped_name}"
            )
        except Exception:
            logger.exception(
                f"Failed to drop Pixeltable view {dropped_name} during LRU eviction; "
                f"registry entry already removed."
            )
        # Re-read: the next loop iteration needs fresh state.
        info = get_video_from_registry(video_id)
        if info is None:
            return


def ensure_domain_view(
    video_id: str,
    domain_context: str,
) -> Optional[str]:
    """Return the view name for (video_id, domain_context), creating on miss.

    * Cache hit: bumps ``last_accessed`` and returns the existing view name.
    * Cache miss: evicts the LRU view if we're at the cap, then synchronously
      builds a fresh one via ``VideoIndexer.create_domain_index``.

    Returns ``None`` if indexing fails — callers should interpret that as
    "domain search unavailable for this query" and carry on with the other
    indexes.
    """
    if not domain_context or not isinstance(domain_context, str):
        return None

    info = get_video_from_registry(video_id)
    if info is None:
        logger.warning(f"ensure_domain_view: video {video_id} not in registry")
        return None

    context_hash = hash_domain_context(domain_context)
    existing = info.domain_views.get(context_hash)
    if existing is not None:
        touch_domain_view(video_id, domain_context)
        return existing["view_name"]

    # Cache miss — evict if needed, then build.
    _evict_lru_if_needed(video_id, _settings.MAX_DOMAIN_VIEWS_PER_VIDEO)

    # Local import to keep this module's import graph light.
    from quadrag.video.indexer import get_indexer

    logger.info(
        f"Lazy-building domain view for {video_id} "
        f"(context='{domain_context}', hash={context_hash})"
    )
    view_name = get_indexer().create_domain_index(video_id, domain_context)
    if view_name is None:
        logger.warning(f"Lazy domain-view build failed for {video_id}")
    return view_name
