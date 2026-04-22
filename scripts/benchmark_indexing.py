#!/usr/bin/env python3
"""Measure how long each of the four indexes takes to build for one video.

Step 9 of the refactor plan: use this before and after the Step-9 concurrency
changes to confirm the swap from custom synchronous UDFs to
``pxt_openai.vision`` produced the expected speedup. The description and
domain indexes should see the biggest wins — they were previously one OpenAI
call per frame in serial; Pixeltable's native async UDFs run them
concurrently with adaptive rate-limit throttling.

Requirements:
    * Backend venv with ``backend/requirements.txt`` installed.
    * ``OPENAI_API_KEY`` and ``GROQ_API_KEY`` set in the environment.
    * An MP4 file at ``scripts/fixtures/benchmark_sample.mp4`` (~1 minute is
      the plan's baseline; any short MP4 works).

Run:
    cd backend && ./.venv/bin/python ../scripts/benchmark_indexing.py

What it prints:
    A table of per-stage wall times plus a total, e.g.::

        stage                   seconds
        ----------------------  -------
        upload + transcode         4.1
        frames view + CLIP index   8.7
        audio view + Whisper      41.3
        description index         23.4
        domain index              22.9
        ------------------------------
        total                    100.4

History tip: keep the output in a small ``docs/benchmarks/YYYY-MM-DD.txt``
per run so regressions stand out in diffs.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = REPO_ROOT / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

# Make Pixeltable's lazy init happy without running FastAPI's top-of-file dance.
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("GOOGLE_API_KEY", "")


def _fatal(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_env(name: str) -> None:
    if not os.environ.get(name):
        _fatal(
            f"{name} is not set. Export it before running this benchmark: "
            f"`export {name}=...`"
        )


def _require_sample_video() -> Path:
    default = REPO_ROOT / "scripts" / "fixtures" / "benchmark_sample.mp4"
    path = Path(os.environ.get("BENCHMARK_VIDEO", str(default)))
    if not path.exists():
        _fatal(
            f"Sample video not found at {path}. Either drop a short MP4 there "
            f"or set BENCHMARK_VIDEO=/path/to/video.mp4"
        )
    return path


@contextmanager
def _timed(label: str, results: list[tuple[str, float]]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        results.append((label, dt))
        print(f"[{dt:7.2f}s] {label}")


def main() -> int:
    _require_env("OPENAI_API_KEY")
    _require_env("GROQ_API_KEY")

    video_path = _require_sample_video()
    video_id = f"bench-{uuid.uuid4().hex[:8]}"
    print(f"Benchmarking with video_id={video_id}, file={video_path}")

    # Imports are deferred so --help works without a backend install.
    import pixeltable as pxt

    pxt.init()

    from quadrag.utils import transcode_video
    from quadrag.video.processor import VideoProcessor
    from quadrag.video.indexer import VideoIndexer

    results: list[tuple[str, float]] = []

    # Copy the sample into the data/videos tree that VideoProcessor expects.
    from quadrag.config import get_settings
    settings = get_settings()
    videos_dir = settings.get_video_dir()
    videos_dir.mkdir(parents=True, exist_ok=True)
    staged_path = videos_dir / f"{video_id}{video_path.suffix}"
    staged_path.write_bytes(video_path.read_bytes())

    with _timed("transcode", results):
        transcoded = transcode_video(str(staged_path))
    # transcode_video writes the H.264 output in-place via rename; use original path.

    processor = VideoProcessor()
    indexer = VideoIndexer()

    with _timed("process_video + register", results):
        processor.process_video(video_id, str(staged_path))

    with _timed("image index (CLIP)", results):
        indexer.create_image_index(video_id)

    with _timed("audio index (Whisper + text embed)", results):
        indexer.create_audio_index(video_id)

    with _timed("description index (vision + text embed)", results):
        indexer.create_description_index(video_id)

    with _timed("domain index (vision + text embed)", results):
        indexer.create_domain_index(video_id, "general video content analysis")

    # Total
    total = sum(dt for _, dt in results)

    print()
    print(f"{'stage':<40} seconds")
    print("-" * 50)
    for label, dt in results:
        print(f"{label:<40} {dt:7.2f}")
    print("-" * 50)
    print(f"{'total':<40} {total:7.2f}")
    print()
    print(f"Video: {video_path} ({staged_path.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
