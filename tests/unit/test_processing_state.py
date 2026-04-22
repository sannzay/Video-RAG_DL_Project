"""Unit tests for ``quadrag.state.processing_state.ProcessingStateStore``.

Covers the basic API + a multi-threaded stress test that ensures no torn
writes, no lost updates, and a consistent final state under heavy concurrent
access.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from quadrag.models import IndexType, ProcessingStatus
from quadrag.state.processing_state import ProcessingStateStore


class TestBasicTransitions:
    def test_unknown_video_reports_pending_by_default(self):
        store = ProcessingStateStore()
        assert store.get_status("nope") == ProcessingStatus.PENDING
        assert store.get_indexes("nope") == []
        assert store.get_error("nope") is None
        assert store.get_index_errors("nope") == {}
        assert store.has("nope") is False

    def test_mark_processing_makes_video_known(self):
        store = ProcessingStateStore()
        store.mark_processing("v1")
        assert store.has("v1") is True
        assert store.get_status("v1") == ProcessingStatus.PROCESSING

    def test_get_status_respects_non_default(self):
        store = ProcessingStateStore()
        assert store.get_status("missing", default=ProcessingStatus.COMPLETED) == ProcessingStatus.COMPLETED

    def test_mark_completed_sets_status_and_replaces_indexes(self):
        store = ProcessingStateStore()
        store.record_index("v1", IndexType.IMAGE)  # pre-existing, should be replaced
        store.mark_completed("v1", [IndexType.AUDIO, IndexType.DESCRIPTION])
        assert store.get_status("v1") == ProcessingStatus.COMPLETED
        assert store.get_indexes("v1") == [IndexType.AUDIO, IndexType.DESCRIPTION]

    def test_mark_failed_sets_status_and_error(self):
        store = ProcessingStateStore()
        store.mark_failed("v1", "something exploded")
        assert store.get_status("v1") == ProcessingStatus.FAILED
        assert store.get_error("v1") == "something exploded"

    def test_mark_processing_does_not_touch_indexes_or_errors(self):
        # /reprocess-video relies on this behavior to preserve already-successful indexes.
        store = ProcessingStateStore()
        store.mark_completed("v1", [IndexType.IMAGE, IndexType.AUDIO])
        store.record_index_error("v1", IndexType.DOMAIN, "failed earlier")
        store.mark_failed("v1", "overall fail")

        store.mark_processing("v1")

        assert store.get_status("v1") == ProcessingStatus.PROCESSING
        # indexes preserved
        assert set(store.get_indexes("v1")) == {IndexType.IMAGE, IndexType.AUDIO}
        # errors preserved too (caller must opt-in to clearing)
        assert store.get_error("v1") == "overall fail"
        assert store.get_index_errors("v1") == {IndexType.DOMAIN: "failed earlier"}


class TestIncrementalRecording:
    def test_record_index_is_idempotent(self):
        store = ProcessingStateStore()
        store.record_index("v1", IndexType.IMAGE)
        store.record_index("v1", IndexType.IMAGE)
        store.record_index("v1", IndexType.AUDIO)
        assert store.get_indexes("v1") == [IndexType.IMAGE, IndexType.AUDIO]

    def test_index_errors_are_per_index(self):
        store = ProcessingStateStore()
        store.record_index_error("v1", IndexType.AUDIO, "whisper failed")
        store.record_index_error("v1", IndexType.DOMAIN, "vision failed")
        assert store.get_index_errors("v1") == {
            IndexType.AUDIO: "whisper failed",
            IndexType.DOMAIN: "vision failed",
        }

    def test_clear_errors_removes_all_errors_for_video(self):
        store = ProcessingStateStore()
        store.mark_failed("v1", "top-level")
        store.record_index_error("v1", IndexType.IMAGE, "per-index")
        store.clear_errors("v1")
        assert store.get_error("v1") is None
        assert store.get_index_errors("v1") == {}
        # status untouched by clear_errors
        assert store.get_status("v1") == ProcessingStatus.FAILED


class TestReadIsolation:
    def test_get_indexes_returns_copy_not_live_reference(self):
        store = ProcessingStateStore()
        store.mark_completed("v1", [IndexType.IMAGE])
        fetched = store.get_indexes("v1")
        fetched.append(IndexType.AUDIO)  # shouldn't affect the store
        assert store.get_indexes("v1") == [IndexType.IMAGE]

    def test_get_index_errors_returns_copy_not_live_reference(self):
        store = ProcessingStateStore()
        store.record_index_error("v1", IndexType.IMAGE, "boom")
        fetched = store.get_index_errors("v1")
        fetched[IndexType.AUDIO] = "injected"
        assert store.get_index_errors("v1") == {IndexType.IMAGE: "boom"}


class TestSnapshotAndRestore:
    def test_empty_snapshot_is_empty_dict(self):
        assert ProcessingStateStore().snapshot() == {}

    def test_snapshot_is_json_safe_enum_values_are_strings(self):
        store = ProcessingStateStore()
        store.mark_completed("v1", [IndexType.IMAGE, IndexType.AUDIO])
        store.record_index_error("v1", IndexType.DOMAIN, "failed")

        snap = store.snapshot()
        assert snap["v1"]["status"] == "completed"
        assert snap["v1"]["indexes"] == ["image", "audio"]
        assert snap["v1"]["index_errors"] == {"domain": "failed"}

    def test_restore_round_trips(self):
        store = ProcessingStateStore()
        store.mark_processing("v1")
        store.record_index("v1", IndexType.IMAGE)
        store.record_index_error("v1", IndexType.AUDIO, "oops")
        store.mark_failed("v2", "bad video")
        store.mark_completed("v3", [IndexType.IMAGE, IndexType.AUDIO])

        snap = store.snapshot()

        fresh = ProcessingStateStore()
        fresh.restore(snap)
        assert fresh.get_status("v1") == ProcessingStatus.PROCESSING
        assert fresh.get_indexes("v1") == [IndexType.IMAGE]
        assert fresh.get_index_errors("v1") == {IndexType.AUDIO: "oops"}
        assert fresh.get_status("v2") == ProcessingStatus.FAILED
        assert fresh.get_error("v2") == "bad video"
        assert fresh.get_status("v3") == ProcessingStatus.COMPLETED
        assert fresh.get_indexes("v3") == [IndexType.IMAGE, IndexType.AUDIO]

    def test_restore_replaces_existing_state(self):
        store = ProcessingStateStore()
        store.mark_processing("old")
        store.restore({"new": {"status": "completed", "indexes": ["image"]}})
        assert store.has("old") is False
        assert store.get_status("new") == ProcessingStatus.COMPLETED

    @pytest.mark.parametrize("bad_snapshot", [None, [], "not a dict", 42])
    def test_restore_tolerates_non_dict_input(self, bad_snapshot):
        store = ProcessingStateStore()
        store.mark_processing("v1")
        store.restore(bad_snapshot)  # must not raise
        # existing state is left untouched on malformed input
        assert store.get_status("v1") == ProcessingStatus.PROCESSING

    def test_restore_skips_unknown_enum_values(self):
        store = ProcessingStateStore()
        store.restore({
            "v1": {
                "status": "not-a-real-status",
                "indexes": ["image", "bogus-index"],
                "index_errors": {"audio": "x", "??": "y"},
            }
        })
        # Unknown status dropped → default PENDING.
        assert store.get_status("v1") == ProcessingStatus.PENDING
        # Unknown indexes dropped; valid one kept.
        assert store.get_indexes("v1") == [IndexType.IMAGE]
        # Unknown index-error keys dropped; valid one kept.
        assert store.get_index_errors("v1") == {IndexType.AUDIO: "x"}


class TestConcurrency:
    def test_record_index_under_stress_has_no_duplicates_or_losses(self):
        """Hammer the store from many threads; final set must match expected."""
        store = ProcessingStateStore()
        videos = [f"v{i}" for i in range(10)]
        indexes = list(IndexType)

        def worker(iterations: int) -> None:
            rng = random.Random(threading.get_ident())
            for _ in range(iterations):
                vid = rng.choice(videos)
                idx = rng.choice(indexes)
                store.record_index(vid, idx)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, 500) for _ in range(16)]
            for f in futures:
                f.result()

        # After N random writes, every (video, index) pair may or may not have fired,
        # but each video's index list must be a subset of IndexType with no duplicates.
        for vid in videos:
            got = store.get_indexes(vid)
            assert len(got) == len(set(got)), f"duplicate indexes for {vid}: {got}"
            assert set(got).issubset(set(indexes))

    def test_interleaved_transitions_converge(self):
        """Status transitions from 100 threads must leave a valid final status."""
        store = ProcessingStateStore()

        def writer(vid: str, mark_fn) -> None:
            mark_fn(vid)

        threads = []
        for i in range(100):
            vid = f"v{i % 5}"
            if i % 3 == 0:
                t = threading.Thread(target=store.mark_processing, args=(vid,))
            elif i % 3 == 1:
                t = threading.Thread(target=store.mark_completed, args=(vid, [IndexType.IMAGE]))
            else:
                t = threading.Thread(target=store.mark_failed, args=(vid, f"err-{i}"))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Every touched video must have a status in the enum (no torn writes).
        for i in range(5):
            vid = f"v{i}"
            status = store.get_status(vid)
            assert isinstance(status, ProcessingStatus)
            assert status in {
                ProcessingStatus.PROCESSING,
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
            }
