"""End-to-end test: POST to /chat after a real video is indexed.

Shares the expensive indexing fixture with the pipeline test via pytest's
fixture cache so we don't pay for two full index builds per test run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def indexed_video(staged_video):
    """Build all four indexes once, reuse across every chat assertion."""
    video_id, staged_path = staged_video

    import pixeltable as pxt
    pxt.init()

    from quadrag.utils import transcode_video
    from quadrag.video.processor import VideoProcessor
    from quadrag.video.indexer import VideoIndexer

    transcode_video(str(staged_path))
    VideoProcessor().process_video(video_id, str(staged_path))

    indexer = VideoIndexer()
    indexer.create_image_index(video_id)
    indexer.create_audio_index(video_id)
    indexer.create_description_index(video_id)
    indexer.create_domain_index(video_id, "general content")
    return video_id


@pytest.fixture
def client():
    """FastAPI test client — runs the app in-process, no real server needed."""
    from api import app  # noqa: WPS433 — late import so heavy deps don't load at collect
    return TestClient(app)


@pytest.mark.integration
@pytest.mark.vcr
def test_chat_returns_grounded_answer(client, indexed_video, cassette_guard):
    """A real question against an indexed video produces an answer + citations."""
    resp = client.post(
        "/chat",
        json={
            "session_id": "integration-session",
            "video_id": indexed_video,
            "query": "What is shown in this video?",
            "domain_context": "general content",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert isinstance(data.get("answer"), str) and data["answer"].strip(), (
        "Groq returned an empty answer"
    )
    assert isinstance(data.get("citations"), list)
    assert len(data["citations"]) >= 1, "Answer has no citations — fusion pipeline is dropping every result"
    assert "grounded" in data, "ChatResponse is missing the Step-11 grounded flag"
    assert isinstance(data["grounded"], bool)

    # Each citation must be structurally well-formed.
    for c in data["citations"]:
        assert set(c.keys()) >= {"content", "timestamp", "similarity", "source"}


@pytest.mark.integration
@pytest.mark.vcr
def test_chat_without_domain_context_still_works(client, indexed_video, cassette_guard):
    """The no-domain-context path must return *some* answer too.

    Regression guard for Step 8's lazy-domain code: /chat with
    ``domain_context=None`` should skip the ensure_domain_view call and still
    return a valid response from the other three indexes.
    """
    resp = client.post(
        "/chat",
        json={
            "session_id": "integration-session",
            "video_id": indexed_video,
            "query": "Describe the video",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("answer"), str) and data["answer"].strip()
