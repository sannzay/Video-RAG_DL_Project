"""Unit tests for the Step-8 multi-domain-view registry API.

Covers the registry-facing primitives that don't require Pixeltable:
``add_domain_view``, ``touch_domain_view``, ``drop_domain_view``,
``lru_domain_view``, ``list_domain_views``, and the hashing helpers.

Pixeltable view creation / eviction side-effects are exercised in the
integration tests (Step 10) with VCR cassettes.
"""

from __future__ import annotations

import time

import pytest

from quadrag.video import registry as registry_mod
from quadrag.video.registry import (
    VideoIndexInfo,
    add_domain_view,
    build_domain_view_name,
    drop_domain_view,
    get_video_from_registry,
    hash_domain_context,
    list_domain_views,
    lru_domain_view,
    touch_domain_view,
)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Swap the on-disk registry path + start from an empty in-memory map.

    Without this, tests clobber each other and any real local state. Cleanup
    restores both to their original values.
    """
    original_file = registry_mod._REGISTRY_FILE
    original_registry = registry_mod._VIDEO_REGISTRY.copy()

    monkeypatch.setattr(registry_mod, "_REGISTRY_FILE", tmp_path / "registry.json")
    registry_mod._VIDEO_REGISTRY.clear()

    yield

    registry_mod._VIDEO_REGISTRY.clear()
    registry_mod._VIDEO_REGISTRY.update(original_registry)
    registry_mod._REGISTRY_FILE = original_file


def _put_video(video_id: str = "v1") -> VideoIndexInfo:
    info = VideoIndexInfo(
        video_id=video_id,
        cache_dir="/tmp/c",
        video_table_name=f"video_{video_id}",
        frames_view_name=f"video_{video_id}_frames",
        audio_view_name=f"video_{video_id}_audio",
        description_view_name=f"video_{video_id}_desc",
    )
    registry_mod._VIDEO_REGISTRY[video_id] = info
    return info


class TestHashAndNaming:
    def test_build_domain_view_name_composes_correctly(self):
        assert build_domain_view_name("vt", "deadbeef") == "vt_domain_deadbeef"

    def test_build_domain_view_name_raises_over_postgres_limit(self):
        prefix = "v" * 60  # way over the 63-char identifier limit once suffixed
        with pytest.raises(ValueError, match="Postgres identifier limit"):
            build_domain_view_name(prefix, "deadbeef")


class TestAddTouchDrop:
    def test_add_domain_view_registers_entry_and_bumps_last_accessed(self, isolated_registry):
        _put_video()
        before = time.time()
        add_domain_view("v1", "cooking", "video_v1_domain_cafebabe")
        info = get_video_from_registry("v1")
        key = hash_domain_context("cooking")
        assert key in info.domain_views
        assert info.domain_views[key]["view_name"] == "video_v1_domain_cafebabe"
        assert info.domain_views[key]["domain_context"] == "cooking"
        assert info.domain_views[key]["last_accessed"] >= before

    def test_add_domain_view_on_unknown_video_is_a_safe_noop(self, isolated_registry):
        # No video registered yet → should not raise.
        add_domain_view("ghost", "ctx", "n")

    def test_touch_bumps_last_accessed_on_existing_entry(self, isolated_registry):
        _put_video()
        add_domain_view("v1", "ctx", "view_ctx")
        key = hash_domain_context("ctx")
        old = get_video_from_registry("v1").domain_views[key]["last_accessed"]

        time.sleep(0.01)
        assert touch_domain_view("v1", "ctx") is True

        new = get_video_from_registry("v1").domain_views[key]["last_accessed"]
        assert new > old

    def test_touch_on_missing_entry_returns_false(self, isolated_registry):
        _put_video()
        assert touch_domain_view("v1", "never-registered") is False

    def test_drop_removes_entry_and_returns_view_name(self, isolated_registry):
        _put_video()
        add_domain_view("v1", "ctx", "view_ctx")
        key = hash_domain_context("ctx")

        dropped = drop_domain_view("v1", key)
        assert dropped == "view_ctx"
        assert key not in get_video_from_registry("v1").domain_views

    def test_drop_of_unknown_hash_returns_none(self, isolated_registry):
        _put_video()
        assert drop_domain_view("v1", "nonexistent") is None


class TestLRUAndListing:
    def test_lru_selects_oldest_last_accessed(self, isolated_registry):
        _put_video()
        add_domain_view("v1", "first",  "view_first")
        time.sleep(0.01)
        add_domain_view("v1", "second", "view_second")
        time.sleep(0.01)
        add_domain_view("v1", "third",  "view_third")

        # bump the middle one so `first` is the true LRU
        touch_domain_view("v1", "second")

        victim = lru_domain_view("v1")
        assert victim is not None
        _, view_name = victim
        assert view_name == "view_first"

    def test_lru_returns_none_when_no_domain_views(self, isolated_registry):
        _put_video()
        assert lru_domain_view("v1") is None

    def test_lru_returns_none_for_unknown_video(self, isolated_registry):
        assert lru_domain_view("missing") is None

    def test_list_is_ordered_most_recent_first(self, isolated_registry):
        _put_video()
        add_domain_view("v1", "a", "view_a")
        time.sleep(0.01)
        add_domain_view("v1", "b", "view_b")
        time.sleep(0.01)
        add_domain_view("v1", "c", "view_c")

        names = [e["view_name"] for e in list_domain_views("v1")]
        assert names == ["view_c", "view_b", "view_a"]


class TestCaseInsensitiveContextLookup:
    def test_differently_cased_contexts_hit_the_same_entry(self, isolated_registry):
        _put_video()
        add_domain_view("v1", "Cooking Techniques", "view_cooking")

        # touch with mismatched case / whitespace — should still resolve.
        assert touch_domain_view("v1", "  cooking techniques ") is True

        info = get_video_from_registry("v1")
        assert info.get_domain_view_name("COOKING techniques") == "view_cooking"
