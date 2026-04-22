"""End-to-end test: ingest a video and verify every index is built + queryable.

This is the single test that catches regressions in the Pixeltable/Whisper/
Vision integration that unit tests can't see. It's expensive — expect 1-2
minutes when recording against real APIs, a few seconds when replaying from
the committed VCR cassette.

Skipped by default (``-m integration`` opts in). Skipped loudly when missing
either (a) the sample video fixture or (b) the cassette on a replay run.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.vcr
def test_full_indexing_pipeline_materializes_all_four_indexes(
    staged_video,
    cassette_guard,
):
    """Drive the full pipeline and assert each index produces real data."""
    video_id, staged_path = staged_video

    # Lazy imports so this test module is still collectable without the
    # backend venv fully installed.
    import pixeltable as pxt
    pxt.init()

    from quadrag.utils import transcode_video
    from quadrag.video.processor import VideoProcessor
    from quadrag.video.indexer import VideoIndexer
    from quadrag.video.registry import get_video_from_registry

    # 1. Transcode → H.264 Main so Pixeltable ingests cleanly.
    transcode_video(str(staged_path))

    # 2. Register the video with Pixeltable and the registry.
    processor = VideoProcessor()
    assert processor.process_video(video_id, str(staged_path)) is True

    indexer = VideoIndexer()

    # 3. Build every index. Each returns a truthy value on success.
    assert indexer.create_image_index(video_id) is True
    assert indexer.create_audio_index(video_id) is True
    assert indexer.create_description_index(video_id) is True
    domain_view_name = indexer.create_domain_index(video_id, "general content")
    assert domain_view_name is not None, "Domain index did not produce a view name"

    # 4. Sanity-check that the views actually have rows.
    info = get_video_from_registry(video_id)
    assert info is not None

    frames = info.frames_view.select().collect()
    assert len(frames) > 0, "Frames view is empty"

    audio = info.audio_view.select(info.audio_view.transcript_text).collect()
    assert len(audio) > 0
    assert any(row["transcript_text"] for row in audio), (
        "Audio index has chunks but every transcript is empty — Whisper call likely failed"
    )

    # Description column lives on frames_view (Step 7 architecture).
    descs = info.frames_view.select(info.frames_view.description).collect()
    assert any(row["description"] for row in descs), "Description column never populated"

    # 5. Semantic search sanity: each text-embedded index must respond to .similarity().
    from quadrag.retrieval.search_engine import VideoSearchEngine

    engine = VideoSearchEngine(video_id, domain_view_name=domain_view_name)
    assert engine.search_audio_index("what is being said") is not None
    assert engine.search_description_index("what do you see") is not None
    # Domain search needs a resolved view — we passed one above.
    domain_hits = engine.search_domain_index("general content overview")
    assert domain_hits is not None
