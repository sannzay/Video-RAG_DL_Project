"""Phase 3 - primary accuracy + additive ablation sweep.

For each of the 5 ablation configurations (``audio_only``, ``image_only``,
``audio_image``, ``plus_desc``, ``full``) we run every QA item in the slice
through retrieval + fusion + generation + MC letter extraction, score against
the Video-MME gold letter, and record per-item and per-config summaries.

Outputs
-------
* ``tests/evaluation/results/eval_runs.jsonl`` - one line per (config, qa)
  with the full record, plus one ``event=config_start`` line per config
  carrying the patched weight vector for provenance.
* ``tests/evaluation/results/eval_summary.csv`` - five rows summarizing each
  configuration's Top-1 accuracy, grounded rate, and 95% bootstrap CI.

Usage
-----
    cd Video-RAG_DL_Project/backend && source .venv/bin/activate
    cd .. && python tests/evaluation/run_eval.py

Idempotent: if ``eval_runs.jsonl`` already contains a (config, question_id)
pair, the script skips it. Pass ``--force`` to re-run every config.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import (
    ABLATION_CONFIGS,
    REPO_ROOT,
    extract_letter,
    format_mc_prompt,
    lens_for_video,
    load_backend_env,
    load_slice,
    patched_weights,
    search_image_index_by_text,
    with_retries,
)

RESULTS_DIR = REPO_ROOT / "tests" / "evaluation" / "results"
RUNS_PATH = RESULTS_DIR / "eval_runs.jsonl"
SUMMARY_PATH = RESULTS_DIR / "eval_summary.csv"


# ----------------------------------------------------------------------------
# Retrieval per config
# ----------------------------------------------------------------------------


def retrieve_for_config(
    video_info,
    search_engine,
    query_text: str,
    config_name: str,
    weights: Dict[str, float],
    top_k_per_index: int = 3,
):
    """Run per-index searches appropriate for a given ablation config.

    Always returns a dict keyed by IndexType. Legs whose weight is zero
    still return [] so the fusion stage sees a consistent schema.
    """
    from quadrag.models import IndexType

    results = {
        IndexType.IMAGE: [],
        IndexType.AUDIO: [],
        IndexType.DESCRIPTION: [],
        IndexType.DOMAIN: [],
    }

    if weights["w_audio"] > 0:
        results[IndexType.AUDIO] = search_engine.search_audio_index(query_text, top_k=top_k_per_index)
    if weights["w_image"] > 0:
        # Custom text->image path since the image index was built without
        # string_embed at register time.
        results[IndexType.IMAGE] = search_image_index_by_text(video_info, query_text, top_k=top_k_per_index)
    if weights["w_desc"] > 0:
        results[IndexType.DESCRIPTION] = search_engine.search_description_index(query_text, top_k=top_k_per_index)
    if weights["w_domain"] > 0 and search_engine.domain_view_name:
        results[IndexType.DOMAIN] = search_engine.search_domain_index(query_text, top_k=top_k_per_index)

    return results


# ----------------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------------


def _load_done_pairs(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "qa_result":
                continue
            done.add((row["config"], row["question_id"]))
    return done


def bootstrap_ci(correct_flags: List[bool], n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI half-widths around the mean accuracy."""
    if not correct_flags:
        return (0.0, 0.0)
    rng = random.Random(0xC0FFEE)
    n = len(correct_flags)
    draws = []
    for _ in range(n_boot):
        sample = [correct_flags[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    low = draws[int(n_boot * (alpha / 2))]
    high = draws[int(n_boot * (1 - alpha / 2))]
    return (low, high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run every (config, qa) pair even if already logged")
    parser.add_argument("--configs", nargs="*",
                        help="Restrict to these config names (default: all 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many QA items per config (smoke-test knob)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip videos not in the registry instead of failing "
                             "(for smoke-testing before the full indexing sweep completes)")
    args = parser.parse_args()

    load_backend_env()
    import os
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    slice_data = load_slice()

    import pixeltable as pxt
    pxt.init()

    # Deferred imports so pixeltable.init() happens first.
    from quadrag.config import get_settings
    from quadrag.models import IndexType
    from quadrag.retrieval.fusion import ResultFusion
    from quadrag.retrieval.search_engine import VideoSearchEngine
    from quadrag.generation.rag_generator import RAGGenerator
    from quadrag.video.registry import get_video_from_registry, hash_domain_context
    from quadrag.video.domain_manager import ensure_domain_view

    settings = get_settings()
    configs = ABLATION_CONFIGS
    if args.configs:
        allowed = set(args.configs)
        configs = [c for c in configs if c[0] in allowed]
        if not configs:
            print(f"ERROR: no configs match {args.configs}", file=sys.stderr)
            return 1

    # Build a map of video_id -> lens (computed once; must match the lens the
    # indexer used so the hashed view name lines up).
    video_by_id = {v["video_id"]: v for v in slice_data["videos"]}
    lens_by_video = {
        v["video_id"]: lens_for_video(v, slice_data["qa"])
        for v in slice_data["videos"]
    }

    # Sanity: every QA points at an indexed video.
    qa_items = slice_data["qa"]
    missing = [qa["question_id"] for qa in qa_items
               if qa["video_id"] not in video_by_id]
    if missing:
        print(f"ERROR: {len(missing)} QA rows reference unknown videos", file=sys.stderr)
        return 1

    # Confirm registry has every video.
    registry_missing = []
    for v in slice_data["videos"]:
        if get_video_from_registry(v["youtube_id"]) is None:
            registry_missing.append(v["youtube_id"])
    if registry_missing:
        if args.skip_missing:
            print(f"WARN: skipping {len(registry_missing)} videos not in registry: "
                  f"{registry_missing}")
            present_ids = {v["video_id"] for v in slice_data["videos"]
                           if v["youtube_id"] not in set(registry_missing)}
            slice_data["videos"] = [v for v in slice_data["videos"]
                                    if v["video_id"] in present_ids]
            slice_data["qa"] = [qa for qa in slice_data["qa"]
                                if qa["video_id"] in present_ids]
            video_by_id = {v["video_id"]: v for v in slice_data["videos"]}
            lens_by_video = {
                v["video_id"]: lens_for_video(v, slice_data["qa"])
                for v in slice_data["videos"]
            }
            qa_items = slice_data["qa"]
        else:
            print(f"ERROR: {len(registry_missing)} videos not in registry: "
                  f"{registry_missing}. Run Phase 2 first.", file=sys.stderr)
            return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done_pairs = set() if args.force else _load_done_pairs(RUNS_PATH)
    if args.force and RUNS_PATH.exists():
        RUNS_PATH.unlink()

    runs_fh = RUNS_PATH.open("a")
    generator = RAGGenerator()

    # Group QA by video so we can create one search engine per video per config.
    qa_by_video: Dict[str, List[dict]] = {}
    for qa in qa_items:
        qa_by_video.setdefault(qa["video_id"], []).append(qa)

    # Ensure every video's domain view is registered. Because Phase 2 built
    # exactly one lens per video and the lens mapping is deterministic, this
    # is a cache lookup in practice.
    resolved_views: Dict[str, Optional[str]] = {}
    for v in slice_data["videos"]:
        yt = v["youtube_id"]
        lens = lens_by_video[v["video_id"]]
        try:
            view_name = ensure_domain_view(yt, lens)
            resolved_views[yt] = view_name
        except Exception as e:
            print(f"WARN: ensure_domain_view failed for {yt} "
                  f"(lens='{lens}'): {e}. Domain leg will be empty.")
            resolved_views[yt] = None

    # ---- main loop ----
    for config_name, weights in configs:
        with patched_weights(**weights) as patched_settings:
            fusion = ResultFusion()
            # Assert monkey-patch took effect.
            for index_type, key in (
                (IndexType.AUDIO, "w_audio"),
                (IndexType.IMAGE, "w_image"),
                (IndexType.DESCRIPTION, "w_desc"),
                (IndexType.DOMAIN, "w_domain"),
            ):
                got = fusion.weights[index_type]
                want = float(weights[key])
                assert abs(got - want) < 1e-6, (
                    f"weight leakage on {config_name}: {index_type} "
                    f"got {got}, want {want}"
                )

            runs_fh.write(json.dumps({
                "event": "config_start",
                "config": config_name,
                "weights": {
                    "audio": patched_settings.WEIGHT_AUDIO,
                    "image": patched_settings.WEIGHT_IMAGE,
                    "description": patched_settings.WEIGHT_DESCRIPTION,
                    "domain": patched_settings.WEIGHT_DOMAIN,
                },
                "ts": time.time(),
            }) + "\n")
            runs_fh.flush()

            print(f"\n=== config: {config_name} ===")
            print(f"  weights: {weights}")

            n_seen = 0
            for vid_id, qa_list in qa_by_video.items():
                vrow = video_by_id[vid_id]
                yt = vrow["youtube_id"]
                view_name = resolved_views.get(yt)
                search_engine = VideoSearchEngine(yt, domain_view_name=view_name)

                for qa in qa_list:
                    if args.limit is not None and n_seen >= args.limit:
                        break
                    n_seen += 1
                    key = (config_name, qa["question_id"])
                    if key in done_pairs:
                        continue

                    t0 = time.perf_counter()
                    try:
                        per_index = retrieve_for_config(
                            search_engine.video_info,
                            search_engine,
                            qa["question"],
                            config_name,
                            weights,
                        )
                        fused = fusion.fuse_results(per_index, top_k=10)
                        prompt = format_mc_prompt(qa)

                        resp = with_retries(
                            lambda: generator.generate_answer(prompt, fused),
                            tries=3,
                        )
                        predicted = extract_letter(resp.answer, qa["options"])
                        correct = predicted == qa["answer"]
                        wall = time.perf_counter() - t0

                        top_sources = [r.source.value if hasattr(r.source, "value") else str(r.source)
                                       for r in fused[:3]]
                        record = {
                            "event": "qa_result",
                            "config": config_name,
                            "question_id": qa["question_id"],
                            "video_id": qa["video_id"],
                            "youtube_id": yt,
                            "duration_bucket": vrow["duration_bucket"],
                            "task_type": qa.get("task_type", ""),
                            "sub_category": qa.get("sub_category", ""),
                            "domain": qa.get("domain", ""),
                            "predicted": predicted,
                            "gold": qa["answer"],
                            "correct": correct,
                            "grounded": resp.grounded,
                            "n_citations": len(resp.citations),
                            "answer_raw": resp.answer,
                            "fused_top_sources": top_sources,
                            "retrieved_counts": {
                                "audio": len(per_index[IndexType.AUDIO]),
                                "image": len(per_index[IndexType.IMAGE]),
                                "description": len(per_index[IndexType.DESCRIPTION]),
                                "domain": len(per_index[IndexType.DOMAIN]),
                            },
                            "wall_clock_sec": wall,
                            "processing_time_server": resp.processing_time,
                            "domain_view_missing": view_name is None,
                        }
                    except Exception as e:
                        record = {
                            "event": "qa_result",
                            "config": config_name,
                            "question_id": qa["question_id"],
                            "video_id": qa["video_id"],
                            "youtube_id": yt,
                            "predicted": None,
                            "gold": qa["answer"],
                            "correct": False,
                            "grounded": False,
                            "error": f"{type(e).__name__}: {e}",
                        }

                    runs_fh.write(json.dumps(record) + "\n")
                    runs_fh.flush()
                    mark = "OK" if record.get("correct") else "..."
                    print(f"  [{config_name}] {qa['question_id']} "
                          f"{mark} pred={record.get('predicted')} gold={qa['answer']} "
                          f"({record.get('wall_clock_sec', 0):.1f}s)")

                if args.limit is not None and n_seen >= args.limit:
                    break

    runs_fh.close()

    # ---- summary ----
    per_config: Dict[str, List[dict]] = {name: [] for name, _ in configs}
    with RUNS_PATH.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "qa_result":
                continue
            if row["config"] in per_config:
                per_config[row["config"]].append(row)

    with SUMMARY_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "n", "top1_accuracy",
                         "grounded_rate", "ci95_low", "ci95_high",
                         "mean_latency_sec"])
        for config_name, _ in configs:
            rows = per_config[config_name]
            if not rows:
                continue
            correct = [bool(r.get("correct")) for r in rows]
            grounded = [bool(r.get("grounded")) for r in rows]
            latencies = [r.get("wall_clock_sec", 0.0) for r in rows if "wall_clock_sec" in r]
            acc = sum(correct) / len(correct)
            grate = sum(grounded) / len(grounded)
            lo, hi = bootstrap_ci(correct)
            mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
            writer.writerow([config_name, len(rows),
                             f"{acc:.4f}", f"{grate:.4f}",
                             f"{lo:.4f}", f"{hi:.4f}",
                             f"{mean_lat:.2f}"])

    print("\n=== summary ===")
    for config_name, _ in configs:
        rows = per_config[config_name]
        if not rows:
            print(f"  {config_name}: (no rows)")
            continue
        correct = sum(1 for r in rows if r.get("correct"))
        grounded = sum(1 for r in rows if r.get("grounded"))
        print(f"  {config_name:>12}: {correct}/{len(rows)} correct "
              f"({correct / len(rows):.1%}), grounded {grounded / len(rows):.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
