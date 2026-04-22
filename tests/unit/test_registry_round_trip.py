"""Round-trip tests for ``VideoIndexInfo`` JSON serialization.

Schema evolution:

* Pre-Step-7: inline ``domain_captions`` / ``domain_context`` JSON fields.
* Step 7: single ``domain_view_name`` pointing at one Pixeltable view.
* Step 8 (current): ``domain_views`` mapping of ``context_hash → entry``,
  one view per (video, domain_context) pair, LRU-evicted at cap.

``from_dict`` must tolerate every historic shape so an upgrade doesn't
crash on existing ``video_registry.json`` files.
"""

from __future__ import annotations

import json

from quadrag.video.registry import (
    VideoIndexInfo,
    hash_domain_context,
)


def _mk_info(**overrides) -> VideoIndexInfo:
    base = {
        "video_id": "abc-123",
        "cache_dir": "/tmp/cache",
        "video_table_name": "video_abc123",
        "frames_view_name": "video_abc123_frames",
        "audio_view_name": "video_abc123_audio",
        "description_view_name": "video_abc123_desc",
    }
    base.update(overrides)
    return VideoIndexInfo(**base)


def test_minimal_round_trip_preserves_names():
    info = _mk_info()
    restored = VideoIndexInfo.from_dict(info.to_dict())
    assert restored.video_id == info.video_id
    assert restored.video_table_name == info.video_table_name
    assert restored.frames_view_name == info.frames_view_name
    assert restored.audio_view_name == info.audio_view_name
    assert restored.description_view_name == info.description_view_name
    assert restored.domain_views == {}


def test_domain_views_round_trip():
    entry = {
        "view_name": "video_abc123_domain_deadbeef",
        "domain_context": "cooking techniques",
        "last_accessed": 1715000000.0,
    }
    info = _mk_info(domain_views={"deadbeef": entry})
    restored = VideoIndexInfo.from_dict(info.to_dict())
    assert "deadbeef" in restored.domain_views
    out = restored.domain_views["deadbeef"]
    assert out["view_name"] == entry["view_name"]
    assert out["domain_context"] == entry["domain_context"]
    assert out["last_accessed"] == entry["last_accessed"]


def test_serialized_form_has_no_legacy_domain_fields():
    """Post-Step-8, nothing under the old ``domain_view_name`` / ``domain_captions`` keys."""
    info = _mk_info()
    raw = info.to_dict()
    assert "domain_captions" not in raw
    assert "domain_context" not in raw
    assert "domain_view_name" not in raw
    assert "domain_views" in raw


def test_to_dict_is_json_compatible():
    info = _mk_info(domain_views={
        "cafebabe": {
            "view_name": "video_abc123_domain_cafebabe",
            "domain_context": "animals",
            "last_accessed": 1.0,
        }
    })
    reloaded = VideoIndexInfo.from_dict(json.loads(json.dumps(info.to_dict())))
    assert "cafebabe" in reloaded.domain_views


def test_pre_step_7_legacy_dicts_are_silently_accepted():
    """Pre-Step-7: captions + context inline, no domain view at all."""
    legacy = {
        "video_id": "pre-7",
        "cache_dir": "/tmp",
        "video_table_name": "vt",
        "frames_view_name": "fv",
        "audio_view_name": "av",
        "description_view_name": "dv",
        "domain_view_name": None,
        "domain_captions": {"1500.0": "old caption"},
        "domain_context": "cooking",
    }
    info = VideoIndexInfo.from_dict(legacy)
    assert info.video_id == "pre-7"
    assert info.domain_views == {}
    # Deprecated fields are not exposed on the object.
    assert not hasattr(info, "domain_captions")
    assert not hasattr(info, "domain_context")


def test_step_7_legacy_single_view_migrates_into_domain_views():
    """Step-7 had one flat ``domain_view_name``; migrate into a ``legacy`` slot."""
    legacy = {
        "video_id": "step-7",
        "cache_dir": "/tmp",
        "video_table_name": "vt",
        "frames_view_name": "fv",
        "audio_view_name": "av",
        "description_view_name": "dv",
        "domain_view_name": "vt_domain",
    }
    info = VideoIndexInfo.from_dict(legacy)
    assert "legacy" in info.domain_views
    assert info.domain_views["legacy"]["view_name"] == "vt_domain"


def test_hash_domain_context_is_deterministic_and_case_insensitive():
    a = hash_domain_context("Cooking Techniques")
    b = hash_domain_context("  cooking techniques  ")
    assert a == b
    assert len(a) == 8


def test_hash_domain_context_collides_only_by_normalization():
    # Distinct normalized strings should yield different hashes.
    assert hash_domain_context("dog training") != hash_domain_context("dog grooming")


def test_get_domain_view_name_uses_hash_keying():
    entry = {
        "view_name": "video_abc123_domain_somehash",
        "domain_context": "cooking",
        "last_accessed": 0.0,
    }
    context_hash = hash_domain_context("cooking")
    info = _mk_info(domain_views={context_hash: entry})
    assert info.get_domain_view_name("Cooking") == entry["view_name"]
    assert info.get_domain_view_name("unrelated") is None


def test_malformed_domain_views_are_dropped_silently():
    """Garbage entries don't crash from_dict — they just don't load."""
    info = VideoIndexInfo.from_dict({
        "video_id": "v",
        "cache_dir": "/tmp",
        "video_table_name": "vt",
        "frames_view_name": "fv",
        "audio_view_name": "av",
        "description_view_name": "dv",
        "domain_views": {
            "good": {"view_name": "ok", "domain_context": "x", "last_accessed": 1.0},
            "bad1": "not a dict",
            "bad2": {"no_view_name_key": True},
        },
    })
    assert set(info.domain_views.keys()) == {"good"}
