"""Thread-safe in-memory store for video processing state.

Replaces the four module-level dicts that used to live in ``backend/api.py``
(``processing_status``, ``processing_errors``, ``video_indexes``,
``_index_errors``) and which were mutated from both the FastAPI event loop and
from daemon processing threads with no locking.

All public methods acquire the store's ``RLock`` and return plain copies of
any mutable data, so callers can't reach in and corrupt internal state.

Persistence:

* When constructed with ``snapshot_path``, the store writes a JSON snapshot
  after every mutation (atomically: tmp file + rename). This survives
  mid-processing crashes and SIGKILL.
* :meth:`load_from_disk` reads a snapshot back at process startup. Missing
  or corrupt files return an empty store with a warning — never crash.

Orphan cleanup:

* When constructed with ``on_fail_cleanup``, that callable is invoked after
  :meth:`mark_failed` runs (outside the lock, so a slow cleanup doesn't
  block readers). Exceptions inside cleanup are logged, never raised.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

from loguru import logger

from quadrag.models import IndexType, ProcessingStatus

PathLike = Union[str, os.PathLike]


class ProcessingStateStore:
    """Thread-safe store for per-video processing status, indexes, and errors."""

    def __init__(
        self,
        snapshot_path: Optional[PathLike] = None,
        on_fail_cleanup: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._status: Dict[str, ProcessingStatus] = {}
        self._indexes: Dict[str, List[IndexType]] = {}
        self._errors: Dict[str, str] = {}
        self._index_errors: Dict[str, Dict[IndexType, str]] = {}
        self._snapshot_path: Optional[Path] = (
            Path(snapshot_path) if snapshot_path is not None else None
        )
        self._on_fail_cleanup: Callable[[str], None] = on_fail_cleanup or (lambda _vid: None)

    # ------------------------------------------------------------------
    # Membership / status transitions
    # ------------------------------------------------------------------
    def has(self, video_id: str) -> bool:
        """Return True if we've ever tracked this video's status."""
        with self._lock:
            return video_id in self._status

    def mark_processing(self, video_id: str) -> None:
        """Set status to PROCESSING. Leaves indexes/errors untouched.

        Callers that want a clean slate (e.g. ``/reprocess-video``) should
        call :meth:`clear_errors` explicitly before this.
        """
        with self._lock:
            self._status[video_id] = ProcessingStatus.PROCESSING
            self._snapshot_to_disk_locked()

    def mark_completed(self, video_id: str, indexes: Iterable[IndexType]) -> None:
        """Set status to COMPLETED and replace the indexes list."""
        with self._lock:
            self._status[video_id] = ProcessingStatus.COMPLETED
            self._indexes[video_id] = list(indexes)
            self._snapshot_to_disk_locked()

    def mark_failed(self, video_id: str, error_msg: str) -> None:
        """Set status to FAILED, record the error, and trigger orphan cleanup.

        Cleanup runs *outside* the lock so a slow Pixeltable drop doesn't
        starve other threads. Cleanup exceptions are logged, never raised —
        failing to clean up shouldn't mask the original failure.
        """
        with self._lock:
            self._status[video_id] = ProcessingStatus.FAILED
            self._errors[video_id] = error_msg
            self._snapshot_to_disk_locked()
        try:
            self._on_fail_cleanup(video_id)
        except Exception:
            logger.exception(f"on_fail_cleanup for {video_id} raised; state already marked FAILED")

    # ------------------------------------------------------------------
    # Fine-grained writes
    # ------------------------------------------------------------------
    def record_index(self, video_id: str, index_type: IndexType) -> None:
        """Append ``index_type`` to the video's indexes list (idempotent)."""
        with self._lock:
            indexes = self._indexes.setdefault(video_id, [])
            if index_type not in indexes:
                indexes.append(index_type)
                self._snapshot_to_disk_locked()

    def record_index_error(
        self, video_id: str, index_type: IndexType, error: str
    ) -> None:
        """Record a per-index error for one video."""
        with self._lock:
            self._index_errors.setdefault(video_id, {})[index_type] = error
            self._snapshot_to_disk_locked()

    def clear_errors(self, video_id: str) -> None:
        """Remove any recorded top-level or per-index errors for this video."""
        with self._lock:
            self._errors.pop(video_id, None)
            self._index_errors.pop(video_id, None)
            self._snapshot_to_disk_locked()

    # ------------------------------------------------------------------
    # Reads (return copies)
    # ------------------------------------------------------------------
    def get_status(
        self,
        video_id: str,
        default: ProcessingStatus = ProcessingStatus.PENDING,
    ) -> ProcessingStatus:
        with self._lock:
            return self._status.get(video_id, default)

    def get_indexes(self, video_id: str) -> List[IndexType]:
        with self._lock:
            return list(self._indexes.get(video_id, []))

    def get_error(self, video_id: str) -> Optional[str]:
        with self._lock:
            return self._errors.get(video_id)

    def get_index_errors(self, video_id: str) -> Dict[IndexType, str]:
        with self._lock:
            return dict(self._index_errors.get(video_id, {}))

    def all_video_ids(self) -> List[str]:
        """Return every video_id we've ever tracked (for debug endpoints)."""
        with self._lock:
            return list(self._status.keys())

    # ------------------------------------------------------------------
    # Persistence hooks (implementation completed in Step 5)
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, object]]:
        """Return a JSON-serializable snapshot of the whole store.

        One entry per video, keyed by video_id. Structure::

            {
                "<video_id>": {
                    "status": "processing" | "completed" | ...,
                    "indexes": ["image", "audio", ...],
                    "error": "<str or None>",
                    "index_errors": {"audio": "...", ...}
                }
            }
        """
        with self._lock:
            return self._snapshot_payload_locked()

    def _snapshot_to_disk_locked(self) -> None:
        """Write the current snapshot to ``self._snapshot_path`` atomically.

        Must be called with ``self._lock`` held. No-op if no path configured.
        Failures are logged but never raised — persistence is best-effort; a
        broken disk must not block in-memory state updates.
        """
        if self._snapshot_path is None:
            return
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._snapshot_payload_locked()
            tmp_path = self._snapshot_path.with_suffix(self._snapshot_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self._snapshot_path)  # atomic on POSIX + Windows
        except Exception:
            logger.exception(f"Failed to persist ProcessingStateStore snapshot to {self._snapshot_path}")

    def _snapshot_payload_locked(self) -> Dict[str, Dict[str, object]]:
        """Same as :meth:`snapshot` but assumes the lock is already held."""
        out: Dict[str, Dict[str, object]] = {}
        video_ids = set(self._status) | set(self._indexes) | set(self._errors) | set(self._index_errors)
        for vid in video_ids:
            status = self._status.get(vid)
            out[vid] = {
                "status": status.value if status is not None else None,
                "indexes": [i.value for i in self._indexes.get(vid, [])],
                "error": self._errors.get(vid),
                "index_errors": {
                    k.value: v for k, v in self._index_errors.get(vid, {}).items()
                },
            }
        return out

    @classmethod
    def load_from_disk(
        cls,
        snapshot_path: PathLike,
        on_fail_cleanup: Optional[Callable[[str], None]] = None,
    ) -> "ProcessingStateStore":
        """Build a store from ``snapshot_path``, falling back to empty on any issue.

        Missing file → empty store (first run).
        Corrupt JSON or unknown schema → empty store + warning (never crash).
        """
        store = cls(snapshot_path=snapshot_path, on_fail_cleanup=on_fail_cleanup)
        path = Path(snapshot_path)
        if not path.exists():
            logger.info(f"No existing snapshot at {path}; starting with empty ProcessingStateStore")
            return store
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Corrupt JSON in {path}; starting with empty ProcessingStateStore")
            return store
        except OSError:
            logger.warning(f"Could not read {path}; starting with empty ProcessingStateStore")
            return store
        store.restore(data)
        logger.info(f"Loaded ProcessingStateStore from {path} ({len(data)} videos)")
        return store

    def restore(self, snapshot: Dict[str, Dict[str, object]]) -> None:
        """Replace in-memory state from a prior snapshot. Tolerant of missing fields."""
        if not isinstance(snapshot, dict):
            logger.warning("ProcessingStateStore.restore: snapshot is not a dict; ignoring")
            return
        with self._lock:
            self._status.clear()
            self._indexes.clear()
            self._errors.clear()
            self._index_errors.clear()

            for vid, entry in snapshot.items():
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping malformed snapshot entry for {vid}")
                    continue

                raw_status = entry.get("status")
                if raw_status:
                    try:
                        self._status[vid] = ProcessingStatus(raw_status)
                    except ValueError:
                        logger.warning(f"Unknown status '{raw_status}' for {vid}; dropping")

                raw_indexes = entry.get("indexes") or []
                indexes: List[IndexType] = []
                for raw in raw_indexes:
                    try:
                        indexes.append(IndexType(raw))
                    except ValueError:
                        logger.warning(f"Unknown index '{raw}' for {vid}; dropping")
                if indexes:
                    self._indexes[vid] = indexes

                err = entry.get("error")
                if err:
                    self._errors[vid] = str(err)

                raw_index_errors = entry.get("index_errors") or {}
                if isinstance(raw_index_errors, dict) and raw_index_errors:
                    idx_errs: Dict[IndexType, str] = {}
                    for raw, msg in raw_index_errors.items():
                        try:
                            idx_errs[IndexType(raw)] = str(msg)
                        except ValueError:
                            logger.warning(f"Unknown index '{raw}' for {vid} errors; dropping")
                    if idx_errs:
                        self._index_errors[vid] = idx_errs
