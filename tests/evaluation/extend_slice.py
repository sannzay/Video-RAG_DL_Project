"""Extend the existing slice by another ``PER_BUCKET`` videos per duration
bucket (5 short + 5 medium + 5 long), and rewrite ``slice.json`` to include
both the old and the new videos and their QA items.

The extension is sampled deterministically from the remaining Video-MME
videos that have a local MP4 present, excluding anything already in the
current slice. All other fields (duration, lens, mp4_path) are populated
the same way as in ``prep_videomme.py``.

Usage:
    cd Video-RAG_DL_Project && python tests/evaluation/extend_slice.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from _common import REPO_ROOT, load_slice

VIDEO_RAG_ROOT = Path("/Users/sanju/Documents/code/intro/Video_RAG")
QA_PATH = VIDEO_RAG_ROOT / "videomme_subset_qa.jsonl"
VIDEOS_CSV = VIDEO_RAG_ROOT / "videomme_subset_videos.csv"
VIDEO_DIR = VIDEO_RAG_ROOT / "videorag"

OUTPUT_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "slice.json"
BACKUP_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "slice_15.json"

SEED = 43   # different from 42 used in prep_videomme.py so we get different draws
PER_BUCKET_ADD = 5
BUCKETS = ("short", "medium", "long")


def _fatal(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _youtube_id_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    path = parsed.path.strip("/")
    if path and "/" not in path and len(path) == 11:
        return path
    if parsed.netloc.endswith("youtu.be") and path:
        return path.split("/")[0]
    return None


def _resolve_video_path(youtube_id: str) -> Optional[Path]:
    for cand in (VIDEO_DIR / f"{youtube_id}.mp4",
                 VIDEO_DIR / "mp4_files" / f"{youtube_id}.mp4",
                 VIDEO_DIR / f"{youtube_id}.webm"):
        if cand.exists():
            return cand
    return None


def _transcode_webm_to_mp4(webm: Path) -> Path:
    mp4 = webm.with_suffix(".mp4")
    if mp4.exists() and mp4.stat().st_size > 0:
        return mp4
    print(f"  transcoding {webm.name} -> {mp4.name}")
    subprocess.run(
        ["ffmpeg", "-i", str(webm), "-c:v", "libx264", "-profile:v", "main",
         "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", "-y", str(mp4)],
        capture_output=True, check=True,
    )
    return mp4


def _duration_sec(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def load_qa_rows() -> List[dict]:
    if not QA_PATH.exists():
        _fatal(f"QA file missing: {QA_PATH}")
    rows = []
    with QA_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_videos_csv() -> List[dict]:
    if not VIDEOS_CSV.exists():
        _fatal(f"videos CSV missing: {VIDEOS_CSV}")
    rows = []
    with VIDEOS_CSV.open() as fh:
        header = fh.readline().strip().split(",")
        assert header == ["video_id", "duration", "url"], f"unexpected header: {header}"
        for line in fh:
            parts = line.strip().split(",", 2)
            if len(parts) != 3:
                continue
            rows.append({"video_id": parts[0], "duration": parts[1], "url": parts[2]})
    return rows


def main() -> int:
    current = load_slice()
    current_ids = {v["video_id"] for v in current["videos"]}
    current_youtube_ids = {v["youtube_id"] for v in current["videos"]}
    print(f"Current slice has {len(current_ids)} videos, "
          f"{len(current['qa'])} QA items")

    # Archive the 15-video version before overwriting.
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(json.dumps(current, indent=2))
        print(f"Archived existing slice to {BACKUP_PATH}")

    qa_rows = load_qa_rows()
    videos_csv = load_videos_csv()

    # Enrich with local paths + youtube id.
    enriched = []
    for row in videos_csv:
        yt = _youtube_id_from_url(row["url"])
        if yt is None:
            continue
        if row["video_id"] in current_ids or yt in current_youtube_ids:
            continue   # already in the slice
        local = _resolve_video_path(yt)
        if local is None:
            continue
        enriched.append({**row, "youtube_id": yt, "local_path": str(local)})

    print(f"Candidates (not already in slice): {len(enriched)}")

    # Sample PER_BUCKET_ADD per bucket.
    rng = random.Random(SEED)
    picked = []
    for bucket in BUCKETS:
        bucket_rows = [r for r in enriched if r["duration"] == bucket]
        if len(bucket_rows) < PER_BUCKET_ADD:
            _fatal(f"only {len(bucket_rows)} candidates in bucket '{bucket}'; "
                   f"need {PER_BUCKET_ADD}")
        bucket_rows.sort(key=lambda r: r["video_id"])
        rng.shuffle(bucket_rows)
        picked.extend(bucket_rows[:PER_BUCKET_ADD])

    print(f"Adding {len(picked)} new videos (seed={SEED})")

    # Ensure MP4s exist.
    for row in picked:
        local = Path(row["local_path"])
        if local.suffix.lower() == ".webm":
            mp4 = _transcode_webm_to_mp4(local)
            row["local_path"] = str(mp4)

    # Probe durations.
    for row in picked:
        row["duration_sec"] = _duration_sec(Path(row["local_path"]))
        print(f"  {row['youtube_id']}: {row['duration_sec']:.1f}s ({row['duration']})")

    # Filter QA to include the new videos' items, on top of the existing qa.
    new_ids = {row["video_id"] for row in picked}
    new_qa = [qa for qa in qa_rows if qa["video_id"] in new_ids]

    # Validate every new video has QA.
    qa_by_video = {vid: [] for vid in new_ids}
    for qa in new_qa:
        qa_by_video[qa["video_id"]].append(qa["question_id"])
    missing = [vid for vid, lst in qa_by_video.items() if not lst]
    if missing:
        _fatal(f"New videos with no QA: {missing}")

    # Assemble the merged slice.
    merged_videos = list(current["videos"]) + [
        {
            "video_id": r["video_id"],
            "youtube_id": r["youtube_id"],
            "duration_bucket": r["duration"],
            "duration_sec": r["duration_sec"],
            "mp4_path": r["local_path"],
            "n_qa": len(qa_by_video[r["video_id"]]),
        }
        for r in picked
    ]
    merged_qa = list(current["qa"]) + [
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
        for qa in new_qa
    ]

    payload = {
        "seed": SEED,
        "per_bucket": PER_BUCKET_ADD,
        "buckets": list(BUCKETS),
        "n_videos": len(merged_videos),
        "n_qa": len(merged_qa),
        "videos": merged_videos,
        "qa": merged_qa,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote merged slice to {OUTPUT_PATH}")
    print(f"  now {len(merged_videos)} videos, {len(merged_qa)} QA items")
    from collections import Counter
    c = Counter(v["duration_bucket"] for v in merged_videos)
    for bucket in BUCKETS:
        print(f"  {bucket:>6}: {c[bucket]} videos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
