"""Phase 1 - build the 15-video Video-MME evaluation slice.

Reads the QA JSONL and videos CSV the user staged under
``/Users/sanju/Documents/code/intro/Video_RAG/`` and produces
``tests/evaluation/data/slice.json`` with a stratified 15-video subset
(5 short + 5 medium + 5 long), the ~45 QA items that attach to them, and
each clip's measured duration.

Also transcodes any WebM-only subset member to MP4 via ffmpeg so downstream
Pixeltable ingest does not trip on the non-MP4 container.

Usage:
    cd /Users/sanju/Documents/code/intro/Video-RAG_DL_Project
    python tests/evaluation/prep_videomme.py

Idempotent. Re-running rebuilds ``slice.json`` from the same seed.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_RAG_ROOT = Path("/Users/sanju/Documents/code/intro/Video_RAG")
QA_PATH = VIDEO_RAG_ROOT / "videomme_subset_qa.jsonl"
VIDEOS_CSV = VIDEO_RAG_ROOT / "videomme_subset_videos.csv"
VIDEO_DIR = VIDEO_RAG_ROOT / "videorag"

OUTPUT_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "slice.json"

# Deterministic sampling so the subset is stable across re-runs.
SEED = 42
PER_BUCKET = 5
BUCKETS = ("short", "medium", "long")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _fatal(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _youtube_id_from_url(url: str) -> Optional[str]:
    """Pull the 11-character YouTube ID out of a watch URL."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    # Fallback: some entries may use /embed/ or youtu.be/ forms.
    path = parsed.path.strip("/")
    if path and "/" not in path and len(path) == 11:
        return path
    if parsed.netloc.endswith("youtu.be") and path:
        return path.split("/")[0]
    return None


def _resolve_video_path(youtube_id: str) -> Optional[Path]:
    """Find the MP4 (preferred) or WebM file for a YouTube ID.

    Looks both directly under ``videorag/`` and under ``videorag/mp4_files/``
    since the staged dataset has a mix of both.
    """
    candidates = [
        VIDEO_DIR / f"{youtube_id}.mp4",
        VIDEO_DIR / "mp4_files" / f"{youtube_id}.mp4",
        VIDEO_DIR / f"{youtube_id}.webm",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _transcode_webm_to_mp4(webm_path: Path) -> Path:
    """Transcode a WebM to H.264 Main MP4 next to the original.

    Idempotent: skips if the MP4 output is already present and non-empty.
    """
    mp4_path = webm_path.with_suffix(".mp4")
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        print(f"  already transcoded: {mp4_path.name}")
        return mp4_path

    print(f"  transcoding {webm_path.name} -> {mp4_path.name} ...")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(webm_path),
                "-c:v", "libx264",
                "-profile:v", "main",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",
                str(mp4_path),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        _fatal(
            f"ffmpeg failed for {webm_path.name}: "
            f"{e.stderr.decode(errors='ignore')[:500]}"
        )
    if not mp4_path.exists() or mp4_path.stat().st_size == 0:
        _fatal(f"transcode produced empty MP4: {mp4_path}")
    return mp4_path


def _duration_sec(path: Path) -> Optional[float]:
    """Return the video's duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout.strip()
        return float(raw) if raw else None
    except (subprocess.CalledProcessError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------


def load_qa_rows() -> List[dict]:
    """Load the Video-MME QA JSONL file."""
    if not QA_PATH.exists():
        _fatal(f"QA file missing: {QA_PATH}")
    rows = []
    with QA_PATH.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                _fatal(f"malformed JSON at {QA_PATH}:{line_no}: {e}")
    return rows


def load_videos_csv() -> List[dict]:
    """Load the Video-MME videos CSV (expected columns: video_id,duration,url)."""
    if not VIDEOS_CSV.exists():
        _fatal(f"videos CSV missing: {VIDEOS_CSV}")
    rows = []
    with VIDEOS_CSV.open() as fh:
        header = fh.readline().strip().split(",")
        expected = ["video_id", "duration", "url"]
        if header != expected:
            _fatal(f"CSV header mismatch: got {header}, expected {expected}")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) != 3:
                continue
            rows.append({"video_id": parts[0], "duration": parts[1], "url": parts[2]})
    return rows


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


def build_subset(videos_csv: List[dict]) -> List[dict]:
    """Sample a stratified subset of PER_BUCKET videos per duration bucket.

    Only considers videos that (a) have a resolvable local MP4 or WebM, so that
    indexing can proceed, and (b) expose a parseable YouTube ID in the URL.
    """
    rng = random.Random(SEED)
    picked: List[dict] = []

    # Attach youtube_id and local path to every CSV row up-front.
    enriched: List[dict] = []
    for row in videos_csv:
        yt = _youtube_id_from_url(row["url"])
        if yt is None:
            continue
        local = _resolve_video_path(yt)
        if local is None:
            continue
        enriched.append({
            **row,
            "youtube_id": yt,
            "local_path": str(local),
        })

    print(f"enriched {len(enriched)} of {len(videos_csv)} CSV rows "
          f"with resolvable local files")

    # Stratify by duration.
    for bucket in BUCKETS:
        bucket_rows = [r for r in enriched if r["duration"] == bucket]
        if len(bucket_rows) < PER_BUCKET:
            _fatal(
                f"only {len(bucket_rows)} videos available in bucket "
                f"'{bucket}' (need {PER_BUCKET}). Add more data or relax the "
                f"PER_BUCKET constant."
            )
        bucket_rows.sort(key=lambda r: r["video_id"])  # stable base order
        rng.shuffle(bucket_rows)
        picked.extend(bucket_rows[:PER_BUCKET])

    return picked


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    print(f"Loading QA data from {QA_PATH} ...")
    qa_rows = load_qa_rows()
    print(f"  loaded {len(qa_rows)} QA items")

    print(f"Loading videos CSV from {VIDEOS_CSV} ...")
    videos_csv = load_videos_csv()
    print(f"  loaded {len(videos_csv)} video rows")

    # Build the subset.
    print(f"\nBuilding stratified {PER_BUCKET * len(BUCKETS)}-video subset "
          f"(seed={SEED}) ...")
    subset = build_subset(videos_csv)
    print(f"  picked {len(subset)} videos "
          f"({', '.join(sorted(r['duration'] for r in subset))})")

    # For every subset member ensure an MP4 exists; transcode webm if needed.
    print("\nEnsuring every subset video has a local MP4 ...")
    for row in subset:
        local = Path(row["local_path"])
        if local.suffix.lower() == ".webm":
            mp4 = _transcode_webm_to_mp4(local)
            row["local_path"] = str(mp4)
        else:
            print(f"  mp4 already present: {local.name}")

    # ffprobe durations.
    print("\nProbing durations ...")
    for row in subset:
        dur = _duration_sec(Path(row["local_path"]))
        row["duration_sec"] = dur
        if dur is None:
            print(f"  WARN: duration unknown for {row['local_path']}")
        else:
            print(f"  {row['youtube_id']}: {dur:.1f}s  ({row['duration']})")

    # Filter QA to the subset.
    subset_ids = {row["video_id"] for row in subset}
    qa_slice = [qa for qa in qa_rows if qa.get("video_id") in subset_ids]

    # Validate: every picked video should have at least one QA item.
    qa_by_video = {vid: [] for vid in subset_ids}
    for qa in qa_slice:
        qa_by_video[qa["video_id"]].append(qa["question_id"])
    missing = [vid for vid, lst in qa_by_video.items() if not lst]
    if missing:
        _fatal(
            f"Video(s) without QA items: {missing}. Refusing to write a slice "
            f"that cannot be evaluated on every member."
        )

    print(f"\nFiltered QA to {len(qa_slice)} items across "
          f"{len(qa_by_video)} videos (avg "
          f"{len(qa_slice) / max(len(qa_by_video), 1):.1f} QA/video)")

    # Assemble the output.
    payload = {
        "seed": SEED,
        "per_bucket": PER_BUCKET,
        "buckets": list(BUCKETS),
        "n_videos": len(subset),
        "n_qa": len(qa_slice),
        "videos": [
            {
                "video_id": row["video_id"],
                "youtube_id": row["youtube_id"],
                "duration_bucket": row["duration"],
                "duration_sec": row["duration_sec"],
                "mp4_path": row["local_path"],
                "n_qa": len(qa_by_video[row["video_id"]]),
            }
            for row in subset
        ],
        "qa": [
            {
                "question_id": qa["question_id"],
                "video_id": qa["video_id"],
                "question": qa["question"],
                "options": qa["options"],
                "answer": qa["answer"],
                "task_type": qa.get("task_type", ""),
                "sub_category": qa.get("sub_category", ""),
                "domain": qa.get("domain", ""),
                "duration_bucket": qa.get("duration", ""),
            }
            for qa in qa_slice
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote slice to {OUTPUT_PATH}")

    # Short summary.
    print("\nSlice summary:")
    for bucket in BUCKETS:
        b_videos = [v for v in payload["videos"] if v["duration_bucket"] == bucket]
        b_qa = sum(v["n_qa"] for v in b_videos)
        b_dur_mean = sum(v["duration_sec"] or 0 for v in b_videos) / max(len(b_videos), 1)
        print(f"  {bucket:>6}: {len(b_videos)} videos, {b_qa} QA, "
              f"mean duration {b_dur_mean:.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
