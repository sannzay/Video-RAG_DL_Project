"""Phase 2b - rebuild each video's domain view under the question-informed
lens produced by ``prep_question_lens.py``.

The original indexing sweep built a lens view per video using a
``sub_category``-derived phrase. Those views remain in the registry and are
kept for backward compatibility, but the eval now targets the new lens.
Because ``MAX_DOMAIN_VIEWS_PER_VIDEO`` is 5 and we only have 2 per video
(the old one + the new one), no LRU eviction happens here.

Wall-clock: ~30 s per video. 15 videos = ~7 min plus a small API spend.

Usage:
    cd Video-RAG_DL_Project/backend && source .venv/bin/activate
    cd .. && python tests/evaluation/rebuild_domain_views.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

from _common import REPO_ROOT, load_backend_env, load_slice

LENSES_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "question_lenses.json"
RESULTS_DIR = REPO_ROOT / "tests" / "evaluation" / "results"
REBUILD_LOG = RESULTS_DIR / "domain_rebuild.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Drop any existing lens view for the same lens "
                             "hash before rebuilding (use this after changing "
                             "the per-frame domain caption prompt).")
    args = parser.parse_args()

    load_backend_env()
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    if not LENSES_PATH.exists():
        print(f"ERROR: {LENSES_PATH} missing. Run prep_question_lens.py first.",
              file=sys.stderr)
        return 1

    lenses: Dict[str, dict] = json.loads(LENSES_PATH.read_text())
    slice_data = load_slice()

    import pixeltable as pxt
    pxt.init()

    from quadrag.video.indexer import get_indexer
    from quadrag.video.registry import (
        get_video_from_registry, hash_domain_context, drop_domain_view,
    )

    indexer = get_indexer()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = REBUILD_LOG.open("w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["youtube_id", "lens", "lens_hash", "status", "wall_clock_sec", "view_name"])

    results = []
    print(f"Rebuilding lens views for {len(slice_data['videos'])} videos")
    print("-" * 70)
    for v in slice_data["videos"]:
        yt = v["youtube_id"]
        meta = lenses.get(yt)
        if not meta:
            print(f"  {yt}: no lens; skipping")
            writer.writerow([yt, "", "", "no_lens", "", ""])
            continue
        lens = meta["lens"]
        lens_hash = hash_domain_context(lens)

        info = get_video_from_registry(yt)
        if info is None:
            print(f"  {yt}: not in registry; skipping")
            writer.writerow([yt, lens, lens_hash, "not_in_registry", "", ""])
            continue

        # Cache hit? The hash of the new lens already indexes an existing view.
        if lens_hash in info.domain_views:
            existing = info.domain_views[lens_hash]["view_name"]
            if args.force:
                print(f"  {yt}: dropping existing view {existing} (force rebuild)")
                try:
                    pxt.drop_table(existing, force=True, if_not_exists="ignore")
                except Exception as e:
                    print(f"    pxt.drop_table failed: {e} -- continuing anyway")
                try:
                    drop_domain_view(yt, lens_hash)
                except Exception as e:
                    print(f"    drop_domain_view failed: {e} -- continuing anyway")
                info = get_video_from_registry(yt)
            else:
                print(f"  {yt}: lens view already present (hash={lens_hash}); skipping build")
                writer.writerow([yt, lens, lens_hash, "cached", "0.0", existing])
                continue

        print(f"  {yt}: building view under lens {lens!r} (hash={lens_hash})")
        t0 = time.perf_counter()
        try:
            view_name = indexer.create_domain_index(yt, lens)
        except Exception as e:
            wall = time.perf_counter() - t0
            print(f"    FAILED: {type(e).__name__}: {e}")
            writer.writerow([yt, lens, lens_hash, f"failed: {e}", f"{wall:.2f}", ""])
            continue
        wall = time.perf_counter() - t0
        if not view_name:
            writer.writerow([yt, lens, lens_hash, "failed: no view", f"{wall:.2f}", ""])
            print(f"    FAILED: no view returned")
            continue
        writer.writerow([yt, lens, lens_hash, "ok", f"{wall:.2f}", view_name])
        log_fh.flush()
        print(f"    built in {wall:.1f}s -> {view_name}")
        results.append((yt, wall))

    log_fh.close()
    print("\n" + "=" * 70)
    print(f"Rebuilt {len(results)} views; log at {REBUILD_LOG}")
    if results:
        total = sum(w for _, w in results)
        print(f"Total wall-clock: {total:.1f}s (mean {total / len(results):.1f}s per video)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
