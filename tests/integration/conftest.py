"""Integration-test fixtures.

These tests actually drive the full indexing + chat pipeline and hit OpenAI /
Groq. We use ``pytest-recording`` (a ``vcrpy`` wrapper) to record every HTTP
call the first time a test runs with live keys, then replay those cassettes
on every subsequent run — so CI never pays the OpenAI bill.

Setup you need to run these:

* A small MP4 at ``tests/fixtures/sample.mp4`` (~3 s, <200 KB). Not
  committed — see ``tests/README.md`` for what to drop in.
* To *record* cassettes the first time: real ``OPENAI_API_KEY`` and
  ``GROQ_API_KEY`` in the environment, plus ``--record-mode=once``.
* To *replay* cassettes: nothing but the committed ``.yaml`` files in
  ``tests/integration/cassettes/``.

If neither the fixture nor the cassette is present, the tests skip with a
clear message — they never silently "pass".
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
CASSETTE_DIR = REPO_ROOT / "tests" / "integration" / "cassettes"
SAMPLE_VIDEO_PATH = FIXTURES_DIR / "sample.mp4"


# ----------------------------------------------------------------------------
# pytest-recording / VCR configuration
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vcr_config():
    """Tell VCR where to put cassettes and which request details to match on.

    * ``filter_headers``: never record ``Authorization`` — we don't want real
      API keys ending up in committed YAML.
    * ``match_on``: ignore query strings (OpenAI SDK sometimes appends
      request IDs) and bodies (large binary image payloads don't serialize
      usefully and change every run).
    * ``decode_compressed_response``: produce human-readable YAML.
    """
    return {
        "cassette_library_dir": str(CASSETTE_DIR),
        "filter_headers": ["authorization", "x-api-key", "openai-organization"],
        "match_on": ["method", "scheme", "host", "path"],
        "decode_compressed_response": True,
    }


# ----------------------------------------------------------------------------
# Fixtures for the indexing + chat pipeline
# ----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_video_path() -> Path:
    """Skip integration tests gracefully if no fixture video is present."""
    if not SAMPLE_VIDEO_PATH.exists():
        pytest.skip(
            f"Integration tests need a sample MP4 at {SAMPLE_VIDEO_PATH}. "
            "See tests/README.md for how to drop one in."
        )
    return SAMPLE_VIDEO_PATH


@pytest.fixture
def isolated_pixeltable_home(tmp_path, monkeypatch):
    """Run each test against a fresh Pixeltable home so there's no cross-test state.

    Without this the second test run would hit cached computed columns and
    wouldn't replay the same HTTP calls as the recording run did, breaking
    cassette playback.
    """
    pxt_home = tmp_path / "pxt_home"
    pxt_home.mkdir()
    monkeypatch.setenv("PIXELTABLE_HOME", str(pxt_home))
    yield pxt_home
    # cleanup handled automatically by tmp_path, but be defensive:
    shutil.rmtree(pxt_home, ignore_errors=True)


@pytest.fixture
def staged_video(isolated_pixeltable_home, sample_video_path, monkeypatch, tmp_path):
    """Copy the sample into a per-test data dir and register a video_id.

    Returns a ``(video_id, staged_path)`` tuple. Uses ``monkeypatch`` to
    redirect ``settings.get_video_dir()`` to the temp path so uploads don't
    pollute the real ``data/videos/``.
    """
    import uuid

    video_dir = tmp_path / "videos"
    cache_dir = tmp_path / "cache"
    video_dir.mkdir()
    cache_dir.mkdir()

    # Force the settings getters to point inside this test's tmpdir.
    from quadrag.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "VIDEO_DIR", str(video_dir))
    monkeypatch.setattr(settings, "CACHE_DIR", str(cache_dir))

    video_id = f"it-{uuid.uuid4().hex[:8]}"
    staged = video_dir / f"{video_id}.mp4"
    shutil.copy(sample_video_path, staged)
    return video_id, staged


@pytest.fixture
def cassette_guard(request):
    """Fail loudly (not silently skip) if a test is in replay mode with no cassette.

    pytest-recording's default behavior when a cassette is missing is to
    record a new one — which would silently make a real API call and fail
    CI without a clear explanation. This fixture checks beforehand.
    """
    if request.config.getoption("--record-mode", default="none") != "none":
        return  # we're recording; don't block
    cassette = CASSETTE_DIR / f"{request.node.originalname}.yaml"
    if not cassette.exists():
        pytest.skip(
            f"No cassette at {cassette}. Re-record with: "
            f"`pytest tests/integration --record-mode=once -m integration` "
            f"after dropping real API keys into the env."
        )
