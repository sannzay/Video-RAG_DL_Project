"""Phase 2 - index every video in ``data/slice.json`` and time each stage.

Feeds the ``tab:indexing-time`` and Efficiency section of the paper. For each
of the 15 videos this script:

    1. Stages the MP4 into ``settings.get_video_dir()`` under its YouTube ID.
    2. Ensures the file is H.264 Main (Pixeltable rejects High profile).
    3. Registers the video via ``VideoProcessor.process_video``.
    4. Builds the four indexes in order: image, audio, description, domain.
    5. Times each stage with ``time.perf_counter`` and writes
       ``results/indexing_times.csv``.

Per-hour extrapolation for the paper is computed from the recorded CSV in a
second pass below, so that the values in the paper are reproducible from the
artifact rather than a transient print.

Usage:
    cd Video-RAG_DL_Project/backend
    source .venv/bin/activate
    cd ..
    python tests/evaluation/run_indexing_sweep.py

Idempotent for videos already in the registry: their stages are skipped and
the CSV row for each stage records ``wall_clock_sec=0, status=skipped``. Pass
``--force`` to re-index from scratch (after cleaning partial artifacts).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Paths and sys.path dance
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = REPO_ROOT / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

# Load .env before touching the pydantic settings cache so backend picks up
# OPENROUTER_API_KEY from it.
_ENV_FILE = REPO_ROOT / "backend" / ".env"
if _ENV_FILE.exists():
    for raw in _ENV_FILE.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

SLICE_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "slice.json"
RESULTS_PATH = REPO_ROOT / "tests" / "evaluation" / "results" / "indexing_times.csv"
PROVENANCE_PATH = REPO_ROOT / "tests" / "evaluation" / "results" / "indexing_provenance.json"


# ----------------------------------------------------------------------------
# Lens mapping - identical between indexing and query time
# ----------------------------------------------------------------------------

# Maps Video-MME ``sub_category`` to a short analytical lens string that the
# description/domain-caption prompts will operate under. Phase 3 must call the
# same function so the lens hash lines up with the view built here.
_DEFAULT_LENS = "general video content analysis"

SUBCATEGORY_LENS: Dict[str, str] = {
    # Knowledge
    "Humanity & History": "historical and cultural analysis",
    "Literature & Art": "artistic and literary analysis",
    "Biology & Medicine": "biological and medical analysis",
    "Physics": "physics concepts and demonstrations",
    "Chemistry": "chemistry concepts and demonstrations",
    "Astronomy": "astronomical and space phenomena",
    "Geography": "geographic and environmental analysis",
    # Film & Television
    "Documentary": "documentary narrative and factual content",
    "Movie": "cinematic narrative and scene analysis",
    "TV Series": "television scene and character analysis",
    "Cartoon": "animated storytelling and character analysis",
    "News Report": "news reporting and current affairs",
    # Sports Competition
    "Basketball": "basketball play and athletic performance",
    "Football": "football play and athletic performance",
    "Soccer": "soccer play and athletic performance",
    "Athletics": "athletic performance analysis",
    "Esports": "competitive video gaming analysis",
    "Other Sports": "sports and athletic performance",
    # Multilingual
    "Multilingual": "multilingual content analysis",
    # Daily Activity
    "Life Record": "daily-life activities and routines",
    "Fashion": "fashion and style analysis",
    # Artistic Performance
    "Dance": "dance and choreography analysis",
    "Music": "music performance and composition analysis",
    "Acrobatics": "acrobatic and stunt performance",
    "Variety Show": "variety-show performance analysis",
    # Life Tip
    "Handicraft": "handicraft technique demonstration",
    "Cooking": "cooking technique and ingredient analysis",
    "Others (Life Tips)": "practical life tips and demonstrations",
    # Scientific
    "Tech & Engineering": "technology and engineering analysis",
    "BusinessFinance": "business and finance analysis",
}


def lens_for_subcategory(sub_category: str) -> str:
    """Pick a reusable lens phrase per Video-MME sub-category.

    Kept short so the sub-category hash used in ``blake2b`` naming converges
    deterministically between the indexing sweep and the eval sweep.
    """
    if not sub_category:
        return _DEFAULT_LENS
    return SUBCATEGORY_LENS.get(sub_category, _DEFAULT_LENS)


# ----------------------------------------------------------------------------
# Lens-per-video derivation
# ----------------------------------------------------------------------------


def lens_for_video(video_row: dict, qa_rows: List[dict]) -> str:
    """Pick the dominant sub-category across a video's QA set and map it.

    Videos can carry multiple QA items with different ``sub_category`` values
    in Video-MME. We pick the most common one so the lens covers the majority
    of the video's questions; ties fall back to the first sub-category seen.
    """
    from collections import Counter

    subs = [qa.get("sub_category", "") for qa in qa_rows if qa["video_id"] == video_row["video_id"]]
    subs = [s for s in subs if s]
    if not subs:
        return _DEFAULT_LENS
    counter = Counter(subs)
    dominant, _ = counter.most_common(1)[0]
    return lens_for_subcategory(dominant)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------


@contextmanager
def timed(label: str, bucket: List[Tuple[str, float]]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket.append((label, time.perf_counter() - t0))


# ----------------------------------------------------------------------------
# Staging
# ----------------------------------------------------------------------------


def stage_video(src: Path, dst_dir: Path, youtube_id: str) -> Path:
    """Hard-link (or copy) the MP4 into ``dst_dir`` as ``{youtube_id}.mp4``.

    Hard-linking is free on the same filesystem and keeps 250 MB clips from
    bloating disk on repeated runs.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{youtube_id}.mp4"
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def ensure_main_profile(path: Path) -> Path:
    """Transcode to H.264 Main only if the input is already High profile.

    Returns the path to use for downstream indexing. Writes alongside the
    original as ``{stem}_transcoded.mp4`` if a transcode happened.
    """
    from quadrag.utils import validate_video_format, transcode_video

    if validate_video_format(str(path)):
        return path
    return Path(transcode_video(str(path)))


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------


def _existing_row_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add((row["youtube_id"], row["stage"]))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild every video even if already indexed")
    parser.add_argument("--video-ids", nargs="*",
                        help="Restrict sweep to the given YouTube IDs")
    args = parser.parse_args()

    # Sanity: required env vars.
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} is not set. See backend/.env", file=sys.stderr)
            return 1

    # Load the slice.
    if not SLICE_PATH.exists():
        print(f"ERROR: {SLICE_PATH} missing. Run prep_videomme.py first.",
              file=sys.stderr)
        return 1
    slice_data = json.loads(SLICE_PATH.read_text())

    # Optionally filter.
    videos = slice_data["videos"]
    if args.video_ids:
        videos = [v for v in videos if v["youtube_id"] in set(args.video_ids)]
    if not videos:
        print("No videos to process.", file=sys.stderr)
        return 1

    print(f"Indexing {len(videos)} video(s)")
    print("-" * 70)

    # Imports deferred until after env vars are in place.
    import pixeltable as pxt
    pxt.init()

    from quadrag.config import get_settings
    from quadrag.video.processor import VideoProcessor, cleanup_partial_pixeltable_artifacts
    from quadrag.video.indexer import VideoIndexer
    from quadrag.video.registry import video_exists_in_registry

    settings = get_settings()
    videos_dir = settings.get_video_dir()
    processor = VideoProcessor()
    indexer = VideoIndexer()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Preserve prior rows unless --force.
    done_keys = set() if args.force else _existing_row_keys(RESULTS_PATH)
    write_header = not RESULTS_PATH.exists() or args.force
    mode = "w" if args.force else "a"

    out = RESULTS_PATH.open(mode, newline="")
    writer = csv.writer(out)
    if write_header:
        writer.writerow(["video_id", "youtube_id", "duration_bucket",
                         "duration_sec", "stage", "wall_clock_sec",
                         "status", "lens"])

    provenance: Dict[str, dict] = {}

    for idx, video in enumerate(videos, 1):
        video_id = video["youtube_id"]
        youtube_id = video["youtube_id"]
        bucket = video["duration_bucket"]
        duration = video["duration_sec"]
        lens = lens_for_video(video, slice_data["qa"])

        print(f"\n[{idx}/{len(videos)}] {youtube_id}  "
              f"({bucket}, {duration:.0f}s, lens='{lens}')")

        # If the registry already has it and we're not forcing, skip.
        if video_exists_in_registry(video_id) and not args.force:
            print(f"  already in registry; skipping all stages")
            for stage in ("stage_video", "transcode", "process_video",
                          "image_index", "audio_index", "description_index",
                          "domain_index"):
                if (youtube_id, stage) in done_keys:
                    continue
                writer.writerow([video_id, youtube_id, bucket, duration,
                                 stage, 0.0, "skipped", lens])
            out.flush()
            provenance[youtube_id] = {"lens": lens, "status": "skipped"}
            continue

        stage_results: List[Tuple[str, float]] = []
        ok = True

        try:
            # 1. Stage
            with timed("stage_video", stage_results):
                staged = stage_video(Path(video["mp4_path"]), videos_dir, youtube_id)

            # 2. Validate / transcode
            with timed("transcode", stage_results):
                final_path = ensure_main_profile(staged)

            # 3. Register
            with timed("process_video", stage_results):
                processor.process_video(video_id, str(final_path))

            # 4. Image index (CLIP)
            with timed("image_index", stage_results):
                indexer.create_image_index(video_id)

            # 5. Audio index (Whisper + embed)
            with timed("audio_index", stage_results):
                indexer.create_audio_index(video_id)

            # 6. Description index (vision + embed)
            with timed("description_index", stage_results):
                indexer.create_description_index(video_id)

            # 7. Domain index (vision + embed) with the video's lens
            with timed("domain_index", stage_results):
                indexer.create_domain_index(video_id, lens)

        except Exception as e:
            ok = False
            print(f"  FAILED: {type(e).__name__}: {e}")
            try:
                cleanup_partial_pixeltable_artifacts(video_id)
                print("  cleaned up partial pixeltable artifacts")
            except Exception as cleanup_err:
                print(f"  cleanup also failed: {cleanup_err}")

        # Write rows.
        for label, dt in stage_results:
            writer.writerow([video_id, youtube_id, bucket, duration, label,
                             f"{dt:.3f}", "ok" if ok else "failed", lens])
        out.flush()

        provenance[youtube_id] = {
            "lens": lens,
            "status": "ok" if ok else "failed",
            "stages_completed": [label for label, _ in stage_results],
            "total_sec": sum(dt for _, dt in stage_results),
        }

        if ok:
            total = sum(dt for _, dt in stage_results)
            print(f"  done in {total:.1f}s "
                  f"({total / max(duration, 1):.2f} s per video-second)")

    out.close()

    # Write provenance.
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2))

    # Summary.
    ok_count = sum(1 for r in provenance.values() if r["status"] == "ok")
    skipped = sum(1 for r in provenance.values() if r["status"] == "skipped")
    failed = sum(1 for r in provenance.values() if r["status"] == "failed")
    print("\n" + "=" * 70)
    print(f"Sweep complete: {ok_count} indexed, {skipped} skipped, {failed} failed")
    print(f"Per-stage CSV:  {RESULTS_PATH}")
    print(f"Provenance:     {PROVENANCE_PATH}")

    # Per-hour extrapolation (informational; paper reads from CSV).
    total_video_sec = sum(v["duration_sec"] for v in videos)
    with RESULTS_PATH.open() as fh:
        reader = csv.DictReader(fh)
        per_stage: Dict[str, float] = {}
        for row in reader:
            if row["status"] != "ok":
                continue
            per_stage[row["stage"]] = per_stage.get(row["stage"], 0.0) + float(row["wall_clock_sec"])
    if total_video_sec > 0:
        print(f"\nPer-hour-of-video extrapolation "
              f"(total {total_video_sec / 60:.1f} min of video):")
        for stage in ("transcode", "process_video", "image_index",
                      "audio_index", "description_index", "domain_index"):
            if stage in per_stage:
                per_hour = per_stage[stage] / total_video_sec * 3600
                print(f"  {stage:>22}: {per_hour:7.1f} s / hr of video")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
