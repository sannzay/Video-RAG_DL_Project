"""Tests for ``ProcessingStateStore`` disk persistence.

Covers:
* writes hit the filesystem on every mutation
* corrupt / missing snapshots recover to an empty store (never crash)
* the on_fail_cleanup hook fires after ``mark_failed`` and is resilient to
  its own exceptions
"""

from __future__ import annotations

import json

import pytest

from quadrag.models import IndexType, ProcessingStatus
from quadrag.state.processing_state import ProcessingStateStore


class TestDiskRoundTrip:
    def test_writes_are_persisted_on_every_mutation(self, tmp_path):
        snap = tmp_path / "state.json"
        store = ProcessingStateStore(snapshot_path=snap)

        store.mark_processing("v1")
        assert snap.exists()

        store.record_index("v1", IndexType.IMAGE)
        data = json.loads(snap.read_text())
        assert data["v1"]["status"] == "processing"
        assert data["v1"]["indexes"] == ["image"]

        store.mark_completed("v1", [IndexType.IMAGE, IndexType.AUDIO])
        data = json.loads(snap.read_text())
        assert data["v1"]["status"] == "completed"
        assert data["v1"]["indexes"] == ["image", "audio"]

    def test_load_from_disk_restores_state(self, tmp_path):
        snap = tmp_path / "state.json"

        original = ProcessingStateStore(snapshot_path=snap)
        original.mark_processing("v1")
        original.record_index("v1", IndexType.AUDIO)
        original.record_index_error("v1", IndexType.IMAGE, "no frames")
        original.mark_failed("v2", "bad video")  # cleanup is a no-op by default

        restored = ProcessingStateStore.load_from_disk(snap)
        assert restored.get_status("v1") == ProcessingStatus.PROCESSING
        assert restored.get_indexes("v1") == [IndexType.AUDIO]
        assert restored.get_index_errors("v1") == {IndexType.IMAGE: "no frames"}
        assert restored.get_status("v2") == ProcessingStatus.FAILED
        assert restored.get_error("v2") == "bad video"

    def test_restored_store_continues_writing_to_same_file(self, tmp_path):
        snap = tmp_path / "state.json"

        first = ProcessingStateStore(snapshot_path=snap)
        first.mark_completed("v1", [IndexType.IMAGE])

        second = ProcessingStateStore.load_from_disk(snap)
        second.mark_completed("v2", [IndexType.AUDIO])

        data = json.loads(snap.read_text())
        assert set(data.keys()) == {"v1", "v2"}

    def test_write_is_atomic_no_tmp_file_left_behind(self, tmp_path):
        snap = tmp_path / "state.json"
        store = ProcessingStateStore(snapshot_path=snap)
        store.mark_processing("v1")

        siblings = list(tmp_path.iterdir())
        # Should be exactly one file: the JSON snapshot. No leftover .tmp.
        assert [p.name for p in siblings] == ["state.json"]


class TestCorruptAndMissing:
    def test_missing_file_returns_empty_store(self, tmp_path):
        snap = tmp_path / "does-not-exist.json"
        store = ProcessingStateStore.load_from_disk(snap)
        assert store.snapshot() == {}

    def test_corrupt_json_returns_empty_store_without_raising(self, tmp_path):
        snap = tmp_path / "state.json"
        snap.write_text("{this is not valid json")
        store = ProcessingStateStore.load_from_disk(snap)
        assert store.snapshot() == {}

    def test_after_corrupt_load_mutations_overwrite_the_bad_file(self, tmp_path):
        snap = tmp_path / "state.json"
        snap.write_text("not json")
        store = ProcessingStateStore.load_from_disk(snap)
        store.mark_completed("v1", [IndexType.IMAGE])

        data = json.loads(snap.read_text())
        assert data["v1"]["status"] == "completed"


class TestFailureCleanupCallback:
    def test_cleanup_invoked_after_mark_failed(self, tmp_path):
        calls = []
        store = ProcessingStateStore(
            snapshot_path=tmp_path / "state.json",
            on_fail_cleanup=calls.append,
        )
        store.mark_failed("v1", "boom")
        assert calls == ["v1"]

    def test_cleanup_not_invoked_on_other_transitions(self, tmp_path):
        calls = []
        store = ProcessingStateStore(
            snapshot_path=tmp_path / "state.json",
            on_fail_cleanup=calls.append,
        )
        store.mark_processing("v1")
        store.mark_completed("v1", [IndexType.IMAGE])
        store.record_index_error("v1", IndexType.AUDIO, "x")
        assert calls == []

    def test_cleanup_exception_does_not_leak_or_undo_state_change(self, tmp_path):
        def bad_cleanup(vid):
            raise RuntimeError("disk on fire")

        store = ProcessingStateStore(
            snapshot_path=tmp_path / "state.json",
            on_fail_cleanup=bad_cleanup,
        )
        # mark_failed must not raise even though cleanup blows up.
        store.mark_failed("v1", "index failed")
        assert store.get_status("v1") == ProcessingStatus.FAILED
        assert store.get_error("v1") == "index failed"


class TestInMemoryStillWorks:
    """Regression: store with no snapshot_path behaves exactly like before."""

    def test_no_snapshot_path_no_disk_writes(self, tmp_path, monkeypatch):
        store = ProcessingStateStore()  # no snapshot_path
        store.mark_processing("v1")
        store.mark_completed("v1", [IndexType.IMAGE])
        # Nothing written under tmp_path since no path was provided.
        assert list(tmp_path.iterdir()) == []
