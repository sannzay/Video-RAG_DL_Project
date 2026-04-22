"""Video index registry for tracking processed videos.

Registry schema evolution:

* Pre-Step-7: single ``domain_view_name`` field *plus* ``domain_captions`` and
  ``domain_context`` stored inline (the JSON-backed pseudo-index).
* Step 7: single ``domain_view_name`` field pointing at one Pixeltable view.
* Step 8 (current): multi-context support via a ``domain_views`` dict keyed
  by an 8-char blake2b hash of the normalized domain context. Each entry
  carries the view name, the original context string (for UI/debug), and
  a ``last_accessed`` timestamp used for LRU eviction.

``from_dict`` silently tolerates every older shape — deployments don't crash
after an upgrade. Legacy single-view entries are carried forward into the new
mapping under a ``"legacy"`` key (the original domain_context isn't known at
load time, so the view is opaquely reusable only by context hash on the next
search).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import pixeltable as pxt
from loguru import logger

# Global registry: _REGISTRY_LOCK guards every read AND write of _VIDEO_REGISTRY
# and the on-disk JSON. Both FastAPI async endpoints and daemon threads touch
# this, so access must be serialized.
_VIDEO_REGISTRY: Dict[str, "VideoIndexInfo"] = {}
_REGISTRY_LOCK = threading.RLock()
_REGISTRY_FILE = Path("../data/cache/video_registry.json")

# Postgres identifier limit. A view name ending in "_domain_{8-char hash}"
# leaves ~54 chars for the prefix — comfortably enough for our naming scheme.
_MAX_VIEW_NAME_LEN = 63


class DomainViewEntry(TypedDict):
    """One entry in ``VideoIndexInfo.domain_views``."""

    view_name: str
    domain_context: str  # original human-readable string
    last_accessed: float  # unix ts; bumped on every search hit, used for LRU


def hash_domain_context(domain_context: str) -> str:
    """8-char hex hash of a domain context, after lowercasing + stripping.

    Collisions would need two contexts that normalize to the same bytes; for
    user-supplied strings that's negligible. Short hash keeps view names
    under the Postgres 63-char identifier limit.
    """
    normalized = (domain_context or "").strip().lower().encode("utf-8")
    return hashlib.blake2b(normalized, digest_size=4).hexdigest()


def build_domain_view_name(video_table_name: str, context_hash: str) -> str:
    """Compose a view name and enforce the Postgres identifier limit."""
    name = f"{video_table_name}_domain_{context_hash}"
    if len(name) >= _MAX_VIEW_NAME_LEN:
        raise ValueError(
            f"Domain view name '{name}' exceeds Postgres identifier limit of "
            f"{_MAX_VIEW_NAME_LEN} chars. Shorten the video table prefix."
        )
    return name


class VideoIndexInfo:
    """Information about a video's Pixeltable indexes.

    Post-Step-8: the Domain Index is a mapping of context-hash → view-entry,
    so one video can have several per-domain-context views simultaneously
    (subject to LRU eviction at :data:`settings.MAX_DOMAIN_VIEWS_PER_VIDEO`).
    """

    def __init__(
        self,
        video_id: str,
        cache_dir: str,
        video_table_name: str,
        frames_view_name: str,
        audio_view_name: str,
        description_view_name: str,
        domain_views: Optional[Dict[str, DomainViewEntry]] = None,
    ):
        self.video_id = video_id
        self.cache_dir = cache_dir
        self.video_table_name = video_table_name
        self.frames_view_name = frames_view_name
        self.audio_view_name = audio_view_name
        self.description_view_name = description_view_name
        self.domain_views: Dict[str, DomainViewEntry] = dict(domain_views or {})

    @property
    def video_table(self):
        return pxt.get_table(self.video_table_name)

    @property
    def frames_view(self):
        return pxt.get_table(self.frames_view_name)

    @property
    def audio_view(self):
        return pxt.get_table(self.audio_view_name)

    @property
    def description_view(self):
        return pxt.get_table(self.description_view_name)

    def get_domain_view_name(self, domain_context: str) -> Optional[str]:
        """Return the view name for this context, or None if not registered."""
        entry = self.domain_views.get(hash_domain_context(domain_context))
        return entry["view_name"] if entry else None

    def get_domain_view(self, domain_context: str):
        """Return the Pixeltable view handle for this context, or None."""
        name = self.get_domain_view_name(domain_context)
        return pxt.get_table(name) if name else None

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "cache_dir": self.cache_dir,
            "video_table_name": self.video_table_name,
            "frames_view_name": self.frames_view_name,
            "audio_view_name": self.audio_view_name,
            "description_view_name": self.description_view_name,
            "domain_views": dict(self.domain_views),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoIndexInfo":
        """Build from a dict, tolerant of every historic registry schema.

        * Pre-Step-7 keys (``domain_captions`` / ``domain_context``): silently
          dropped — that content now lives in Pixeltable, not JSON.
        * Step-7 key (``domain_view_name``): carried forward into
          ``domain_views`` under a ``"legacy"`` hash with a placeholder
          timestamp, so the existing Pixeltable view remains reachable by a
          later admin migration. On first /chat with a real context the new
          path will build a properly-keyed entry and the legacy one eventually
          ages out via LRU.
        * Step-8 key (``domain_views``): used as-is.
        """
        core_known = {
            "video_id",
            "cache_dir",
            "video_table_name",
            "frames_view_name",
            "audio_view_name",
            "description_view_name",
        }
        kwargs = {k: v for k, v in data.items() if k in core_known}

        raw_views = data.get("domain_views")
        domain_views: Dict[str, DomainViewEntry] = {}
        if isinstance(raw_views, dict):
            for h, entry in raw_views.items():
                if not isinstance(entry, dict) or "view_name" not in entry:
                    continue
                domain_views[str(h)] = DomainViewEntry(
                    view_name=str(entry["view_name"]),
                    domain_context=str(entry.get("domain_context", "")),
                    last_accessed=float(entry.get("last_accessed", 0.0)),
                )

        # Step-7 → Step-8 migration: a flat ``domain_view_name`` with no
        # matching ``domain_views`` becomes a single legacy entry.
        legacy_name = data.get("domain_view_name")
        if legacy_name and not domain_views:
            domain_views["legacy"] = DomainViewEntry(
                view_name=str(legacy_name),
                domain_context=str(data.get("domain_context", "") or ""),
                last_accessed=0.0,
            )

        kwargs["domain_views"] = domain_views
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

def add_video_to_registry(
    video_id: str,
    cache_dir: str,
    video_table_name: str,
    frames_view_name: str,
    audio_view_name: str,
    description_view_name: str,
) -> VideoIndexInfo:
    """Add a fresh video entry — no domain views yet."""
    info = VideoIndexInfo(
        video_id=video_id,
        cache_dir=cache_dir,
        video_table_name=video_table_name,
        frames_view_name=frames_view_name,
        audio_view_name=audio_view_name,
        description_view_name=description_view_name,
    )
    with _REGISTRY_LOCK:
        _VIDEO_REGISTRY[video_id] = info
        _save_registry_locked()
    logger.info(f"Added video {video_id} to registry")
    return info


def get_video_from_registry(video_id: str) -> Optional[VideoIndexInfo]:
    with _REGISTRY_LOCK:
        return _VIDEO_REGISTRY.get(video_id)


def video_exists_in_registry(video_id: str) -> bool:
    with _REGISTRY_LOCK:
        return video_id in _VIDEO_REGISTRY


def get_all_videos() -> Dict[str, VideoIndexInfo]:
    """Shallow-copy of the registry mapping."""
    with _REGISTRY_LOCK:
        return _VIDEO_REGISTRY.copy()


# ---------------------------------------------------------------------------
# Domain-view CRUD (multi-context, LRU-evicted)
# ---------------------------------------------------------------------------

def add_domain_view(
    video_id: str,
    domain_context: str,
    view_name: str,
) -> None:
    """Register a new domain view, keyed by the normalized-context hash.

    Callers must drop the Pixeltable view *before* calling this if they're
    replacing an existing entry; this function only touches the registry.
    """
    with _REGISTRY_LOCK:
        info = _VIDEO_REGISTRY.get(video_id)
        if info is None:
            logger.warning(f"add_domain_view: video {video_id} not in registry")
            return
        context_hash = hash_domain_context(domain_context)
        info.domain_views[context_hash] = DomainViewEntry(
            view_name=view_name,
            domain_context=domain_context,
            last_accessed=time.time(),
        )
        _save_registry_locked()
    logger.info(
        f"Registered domain view for {video_id} (context='{domain_context}', "
        f"hash={context_hash}, view={view_name})"
    )


def touch_domain_view(video_id: str, domain_context: str) -> bool:
    """Bump ``last_accessed`` on a known domain view; no-op if absent."""
    with _REGISTRY_LOCK:
        info = _VIDEO_REGISTRY.get(video_id)
        if info is None:
            return False
        entry = info.domain_views.get(hash_domain_context(domain_context))
        if entry is None:
            return False
        entry["last_accessed"] = time.time()
        _save_registry_locked()
        return True


def drop_domain_view(video_id: str, context_hash: str) -> Optional[str]:
    """Remove the registry entry for a (video, context_hash) pair.

    Returns the dropped view name (so the caller can also drop the underlying
    Pixeltable view), or None if nothing was removed.
    """
    with _REGISTRY_LOCK:
        info = _VIDEO_REGISTRY.get(video_id)
        if info is None:
            return None
        entry = info.domain_views.pop(context_hash, None)
        if entry is None:
            return None
        _save_registry_locked()
    logger.info(f"Dropped domain view registry entry for {video_id} (hash={context_hash})")
    return entry["view_name"]


def lru_domain_view(video_id: str) -> Optional[Tuple[str, str]]:
    """Return (context_hash, view_name) of the LRU domain view, or None."""
    with _REGISTRY_LOCK:
        info = _VIDEO_REGISTRY.get(video_id)
        if info is None or not info.domain_views:
            return None
        hash_key = min(
            info.domain_views,
            key=lambda k: info.domain_views[k].get("last_accessed", 0.0),
        )
        return hash_key, info.domain_views[hash_key]["view_name"]


def list_domain_views(video_id: str) -> List[DomainViewEntry]:
    """Sorted-by-recency (most recent first) domain views for a video."""
    with _REGISTRY_LOCK:
        info = _VIDEO_REGISTRY.get(video_id)
        if info is None:
            return []
        return sorted(
            info.domain_views.values(),
            key=lambda e: e.get("last_accessed", 0.0),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_registry_locked() -> None:
    """Atomic persist. Assumes caller holds ``_REGISTRY_LOCK``."""
    try:
        _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _REGISTRY_FILE.with_suffix(_REGISTRY_FILE.suffix + ".tmp")
        data = {vid: info.to_dict() for vid, info in _VIDEO_REGISTRY.items()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(_REGISTRY_FILE)
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")


def _load_registry() -> None:
    """Load registry from disk. Called once at module import."""
    global _VIDEO_REGISTRY
    with _REGISTRY_LOCK:
        try:
            if _REGISTRY_FILE.exists():
                with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _VIDEO_REGISTRY = {
                        vid: VideoIndexInfo.from_dict(info_dict)
                        for vid, info_dict in data.items()
                    }
                logger.info(f"Loaded {len(_VIDEO_REGISTRY)} videos from registry")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            _VIDEO_REGISTRY = {}


# Load registry on module import
_load_registry()
