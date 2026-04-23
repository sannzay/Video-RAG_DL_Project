"""Phase 4 - end-to-end query latency and token-cost sweep.

Picks two archetype videos (one dialogue-heavy, one visual-heavy) already
indexed by Phase 2, then issues N=100 queries against each sampling from the
slice's QA pool (with replacement if fewer than 100 items per video). For
every query we time each pipeline stage independently with ``perf_counter``
and capture the OpenRouter token usage for the chat call.

Outputs
-------
* ``results/query_latency.csv``  - per-query stage times + total.
* ``results/query_latency_summary.csv`` - per-stage mean/median/p95.
* ``results/token_counts.csv``    - per-query prompt / completion tokens
  plus an estimated dollar cost.

Usage
-----
    cd Video-RAG_DL_Project/backend && source .venv/bin/activate
    cd .. && python tests/evaluation/run_query_latency.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import (
    REPO_ROOT,
    format_mc_prompt,
    lens_for_video,
    load_backend_env,
    load_slice,
)

RESULTS_DIR = REPO_ROOT / "tests" / "evaluation" / "results"
LATENCY_CSV = RESULTS_DIR / "query_latency.csv"
LATENCY_SUMMARY_CSV = RESULTS_DIR / "query_latency_summary.csv"
TOKEN_CSV = RESULTS_DIR / "token_counts.csv"


# ----------------------------------------------------------------------------
# OpenRouter price book (dated; matches Llama 3.3 70B Instruct rack rate
# at the time of this eval). Update together with settings.CHAT_MODEL.
# ----------------------------------------------------------------------------

# Reference: openrouter.ai pricing, April 2026 snapshot.
CHAT_MODEL_INPUT_PER_M = 0.13    # USD per 1M input tokens
CHAT_MODEL_OUTPUT_PER_M = 0.39   # USD per 1M output tokens

# OpenAI text-embedding-3-small, same snapshot.
EMBED_INPUT_PER_M = 0.020        # USD per 1M tokens


# ----------------------------------------------------------------------------
# Archetype picker
# ----------------------------------------------------------------------------


DIALOGUE_SUBCATS = {
    "Humanity & History", "Literature & Art", "News Report", "Documentary",
    "Variety Show", "Multilingual", "BusinessFinance",
}

VISUAL_SUBCATS = {
    "Cooking", "Handicraft", "Dance", "Acrobatics", "Basketball", "Football",
    "Soccer", "Athletics", "Fashion", "Others (Life Tips)",
}


def pick_archetype_videos(slice_data: dict) -> Tuple[dict, dict]:
    """Return (dialogue_heavy_video, visual_heavy_video) from the slice.

    Uses the dominant sub-category of each video's QA set to classify.
    Falls back gracefully if no clean match is found.
    """
    qa_items = slice_data["qa"]
    from collections import Counter

    best_dialogue = None
    best_visual = None
    for v in slice_data["videos"]:
        subs = [qa.get("sub_category", "") for qa in qa_items if qa["video_id"] == v["video_id"]]
        if not subs:
            continue
        dominant, _ = Counter(subs).most_common(1)[0]
        if dominant in DIALOGUE_SUBCATS and best_dialogue is None:
            best_dialogue = v
        if dominant in VISUAL_SUBCATS and best_visual is None:
            best_visual = v
        if best_dialogue and best_visual:
            break

    # Fallbacks: any long for dialogue, any short for visual.
    if best_dialogue is None:
        best_dialogue = next((v for v in slice_data["videos"]
                              if v["duration_bucket"] == "long"), slice_data["videos"][0])
    if best_visual is None:
        best_visual = next((v for v in slice_data["videos"]
                            if v["duration_bucket"] == "short"),
                           slice_data["videos"][-1])

    return best_dialogue, best_visual


# ----------------------------------------------------------------------------
# Instrumented one-query runner
# ----------------------------------------------------------------------------


def run_one_query(
    generator,
    search_engine,
    fusion,
    qa: dict,
    apply_grounding_fn,
):
    """Run a single query end-to-end and return the stage times + token usage.

    The upstream ``RAGGenerator.generate_answer`` wraps retrieval + generation
    + grounding as a single function. We replicate its internal flow here so
    that each stage can be timed independently.
    """
    timings: Dict[str, float] = {}

    # --- retrieval ---
    t0 = time.perf_counter()
    per_index = search_engine.search_all_indexes(qa["question"])
    timings["retrieval_sec"] = time.perf_counter() - t0

    # --- fusion + dedup ---
    t0 = time.perf_counter()
    fused = fusion.fuse_results(per_index, top_k=10)
    timings["fusion_sec"] = time.perf_counter() - t0

    # --- LLM call ---
    from quadrag.config import get_settings
    settings = get_settings()
    prompt = generator.build_context_prompt(format_mc_prompt(qa), fused)
    t0 = time.perf_counter()
    response = generator.client.chat.completions.create(
        model=generator.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.GROQ_TEMPERATURE,
        max_tokens=settings.GROQ_MAX_TOKENS,
    )
    timings["generation_sec"] = time.perf_counter() - t0

    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # --- grounding (regex + filter) ---
    t0 = time.perf_counter()
    citations, grounded = apply_grounding_fn(answer, fused)
    timings["grounding_sec"] = time.perf_counter() - t0

    timings["total_sec"] = (
        timings["retrieval_sec"] + timings["fusion_sec"]
        + timings["generation_sec"] + timings["grounding_sec"]
    )
    return timings, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "grounded": grounded,
        "n_citations": len(citations),
        "answer": answer,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Queries per video (default: 100)")
    parser.add_argument("--warm-queries", type=int, default=1,
                        help="How many lead queries to exclude from averages (cold-cache)")
    args = parser.parse_args()

    load_backend_env()
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    slice_data = load_slice()

    import pixeltable as pxt
    pxt.init()

    from quadrag.config import get_settings
    from quadrag.retrieval.fusion import ResultFusion
    from quadrag.retrieval.search_engine import VideoSearchEngine
    from quadrag.generation.rag_generator import RAGGenerator, apply_citation_grounding
    from quadrag.video.registry import get_video_from_registry
    from quadrag.video.domain_manager import ensure_domain_view

    dialogue_video, visual_video = pick_archetype_videos(slice_data)

    print(f"Archetype videos:")
    print(f"  dialogue-heavy: {dialogue_video['youtube_id']} "
          f"({dialogue_video['duration_bucket']}, {dialogue_video['duration_sec']:.0f}s)")
    print(f"  visual-heavy:   {visual_video['youtube_id']} "
          f"({visual_video['duration_bucket']}, {visual_video['duration_sec']:.0f}s)")

    # Both videos must be indexed.
    for v in (dialogue_video, visual_video):
        if get_video_from_registry(v["youtube_id"]) is None:
            print(f"ERROR: {v['youtube_id']} not in registry. Run Phase 2 first.",
                  file=sys.stderr)
            return 1

    qa_items = slice_data["qa"]
    generator = RAGGenerator()
    settings = get_settings()
    fusion = ResultFusion()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latency_fh = LATENCY_CSV.open("w", newline="")
    latency_w = csv.writer(latency_fh)
    latency_w.writerow(["workload", "youtube_id", "qa_index", "question_id",
                        "retrieval_sec", "fusion_sec", "generation_sec",
                        "grounding_sec", "total_sec", "cold"])

    token_fh = TOKEN_CSV.open("w", newline="")
    token_w = csv.writer(token_fh)
    token_w.writerow(["workload", "youtube_id", "qa_index", "question_id",
                      "prompt_tokens", "completion_tokens", "total_tokens",
                      "est_cost_usd"])

    rng = random.Random(0xBEEF)

    def per_workload(label: str, video: dict):
        yt = video["youtube_id"]
        lens = lens_for_video(video, qa_items)
        view_name = ensure_domain_view(yt, lens)
        engine = VideoSearchEngine(yt, domain_view_name=view_name)

        pool = [qa for qa in qa_items if qa["video_id"] == video["video_id"]]
        if not pool:
            print(f"  no QA items for {yt}; skipping")
            return []

        print(f"\n=== {label}: {yt} ({len(pool)} QA in pool, sampling {args.n}) ===")
        rows: List[Tuple[Dict[str, float], dict]] = []
        for i in range(args.n):
            qa = pool[rng.randrange(len(pool))]
            try:
                timings, meta = run_one_query(
                    generator, engine, fusion, qa, apply_citation_grounding
                )
            except Exception as e:
                print(f"  {i:>3}: FAILED ({type(e).__name__}: {e})")
                continue
            cold = i < args.warm_queries
            latency_w.writerow([
                label, yt, i, qa["question_id"],
                f"{timings['retrieval_sec']:.4f}",
                f"{timings['fusion_sec']:.4f}",
                f"{timings['generation_sec']:.4f}",
                f"{timings['grounding_sec']:.4f}",
                f"{timings['total_sec']:.4f}",
                "1" if cold else "0",
            ])
            total_toks = meta["prompt_tokens"] + meta["completion_tokens"]
            cost = (
                meta["prompt_tokens"] / 1e6 * CHAT_MODEL_INPUT_PER_M
                + meta["completion_tokens"] / 1e6 * CHAT_MODEL_OUTPUT_PER_M
            )
            token_w.writerow([
                label, yt, i, qa["question_id"],
                meta["prompt_tokens"], meta["completion_tokens"], total_toks,
                f"{cost:.6f}",
            ])
            latency_fh.flush()
            token_fh.flush()
            if i % 10 == 0:
                print(f"  {i:>3}: total={timings['total_sec']:.2f}s "
                      f"retr={timings['retrieval_sec']:.2f} "
                      f"fuse={timings['fusion_sec']:.3f} "
                      f"gen={timings['generation_sec']:.2f} "
                      f"ground={timings['grounding_sec']:.3f} "
                      f"toks={total_toks}")
            rows.append((timings, meta))
        return rows

    d_rows = per_workload("dialogue_heavy", dialogue_video)
    v_rows = per_workload("visual_heavy", visual_video)

    latency_fh.close()
    token_fh.close()

    # ---- summary ----
    def summarize(label: str, rows: List[Tuple[Dict[str, float], dict]], skip_cold: int):
        warm = rows[skip_cold:]
        if not warm:
            return []
        stages = ("retrieval_sec", "fusion_sec", "generation_sec",
                  "grounding_sec", "total_sec")
        out = []
        for stage in stages:
            vals = [t[stage] for t, _ in warm]
            out.append({
                "workload": label,
                "stage": stage,
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "p95": _p95(vals),
            })
        return out

    def _p95(vals: List[float]) -> float:
        vals = sorted(vals)
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, int(len(vals) * 0.95))
        return vals[idx]

    summary_rows = summarize("dialogue_heavy", d_rows, args.warm_queries) \
        + summarize("visual_heavy", v_rows, args.warm_queries)

    with LATENCY_SUMMARY_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "stage", "n", "mean_sec", "median_sec", "p95_sec"])
        for row in summary_rows:
            w.writerow([
                row["workload"], row["stage"], row["n"],
                f"{row['mean']:.4f}", f"{row['median']:.4f}", f"{row['p95']:.4f}",
            ])

    # Also print cost summaries.
    def cost_summary(label: str, rows: List[Tuple[Dict[str, float], dict]]):
        if not rows:
            return
        prompt_tokens = [r[1]["prompt_tokens"] for r in rows]
        completion_tokens = [r[1]["completion_tokens"] for r in rows]
        n = len(rows)
        total_cost = sum(
            pt / 1e6 * CHAT_MODEL_INPUT_PER_M + ct / 1e6 * CHAT_MODEL_OUTPUT_PER_M
            for pt, ct in zip(prompt_tokens, completion_tokens)
        )
        print(f"\n  {label}: n={n}  "
              f"mean_prompt_toks={statistics.mean(prompt_tokens):.0f}  "
              f"mean_completion_toks={statistics.mean(completion_tokens):.0f}  "
              f"cost/100q=${(total_cost / n * 100):.4f}")

    print("\n=== cost summary ===")
    cost_summary("dialogue_heavy", d_rows)
    cost_summary("visual_heavy", v_rows)

    print(f"\nPer-query latency CSV:  {LATENCY_CSV}")
    print(f"Summary CSV:            {LATENCY_SUMMARY_CSV}")
    print(f"Token counts CSV:       {TOKEN_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
