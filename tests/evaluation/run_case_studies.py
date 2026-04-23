"""Phase 5 - pick three qualitative case studies from the eval run.

Reads ``results/eval_runs.jsonl``, filters for QA items that cleanly separate
one of the four indexes, and records the exact question, answers, and
citations for each archetype so §V.E of the paper can be populated.

Archetypes (from the paper's commitment in §V.E):
    1. Dialogue-heavy     - deciding index = Audio
    2. Purely visual      - deciding index = Description or Image
    3. Lens-dependent     - deciding index = Domain/Lens

Usage:
    python tests/evaluation/run_case_studies.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import (
    REPO_ROOT,
    format_mc_prompt,
    lens_for_video,
    load_backend_env,
    load_slice,
    patched_weights,
    with_retries,
)

RESULTS_DIR = REPO_ROOT / "tests" / "evaluation" / "results"
RUNS_PATH = RESULTS_DIR / "eval_runs.jsonl"
OUTPUT_PATH = RESULTS_DIR / "case_studies.json"

DIALOGUE_TASKS = {"Information Synopsis", "Temporal Reasoning", "Counting"}
VISUAL_TASKS = {"Object Recognition", "Action Recognition", "Scene Understanding",
                "Object Reasoning"}
# Synthesized weights for the "full minus lens" rerun used by archetype 3.
NO_LENS_WEIGHTS = {"w_audio": 0.40, "w_image": 0.27, "w_desc": 0.33, "w_domain": 0.00}


def load_eval_rows() -> List[dict]:
    if not RUNS_PATH.exists():
        raise SystemExit(f"{RUNS_PATH} missing. Run run_eval.py first.")
    rows = []
    with RUNS_PATH.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "qa_result":
                rows.append(row)
    return rows


def index_by_qa_config(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["question_id"], r["config"]): r for r in rows}


def pick_archetype_1(rows: List[dict], idx: Dict[Tuple[str, str], dict]) -> Optional[dict]:
    """Dialogue-heavy: audio_only correct AND image_only wrong."""
    candidates = []
    for r in rows:
        if r["config"] != "audio_only" or not r.get("correct"):
            continue
        image_r = idx.get((r["question_id"], "image_only"))
        if image_r and not image_r.get("correct"):
            candidates.append(r)

    # Prefer task types in DIALOGUE_TASKS, then longer audio retrieval hits.
    candidates.sort(key=lambda r: (
        0 if r.get("task_type") in DIALOGUE_TASKS else 1,
        -r.get("retrieved_counts", {}).get("audio", 0),
    ))
    return candidates[0] if candidates else None


def pick_archetype_2(rows: List[dict], idx: Dict[Tuple[str, str], dict]) -> Optional[dict]:
    """Visual: image_only or plus_desc correct AND audio_only wrong."""
    candidates = []
    for r in rows:
        if r["config"] not in {"image_only", "plus_desc"}:
            continue
        if not r.get("correct"):
            continue
        audio_r = idx.get((r["question_id"], "audio_only"))
        if audio_r and not audio_r.get("correct"):
            candidates.append(r)
    candidates.sort(key=lambda r: (
        0 if r.get("task_type") in VISUAL_TASKS else 1,
        # Prefer image_only over plus_desc since it's more informative.
        0 if r["config"] == "image_only" else 1,
    ))
    return candidates[0] if candidates else None


def pick_archetype_3_candidates(
    rows: List[dict], idx: Dict[Tuple[str, str], dict]
) -> List[dict]:
    """Lens-dependent: full is correct.

    Archetype 3 needs an on-the-fly rerun with lens disabled; we return the
    candidate pool here and caller picks + reruns one.
    """
    candidates = [r for r in rows
                  if r["config"] == "full" and r.get("correct")]
    # Prefer QAs where audio_only AND image_only BOTH were wrong. Those are
    # cases where only the fused answer (with domain) succeeded.
    def rank_key(r):
        audio_wrong = idx.get((r["question_id"], "audio_only"), {}).get("correct") is False
        image_wrong = idx.get((r["question_id"], "image_only"), {}).get("correct") is False
        return (not (audio_wrong and image_wrong), 0)
    candidates.sort(key=rank_key)
    return candidates


def rerun_no_lens(qa: dict, video: dict, slice_qa: List[dict]) -> dict:
    """Run a single QA item under weights that zero-out the lens leg."""
    from quadrag.config import get_settings
    from quadrag.models import IndexType
    from quadrag.retrieval.fusion import ResultFusion
    from quadrag.retrieval.search_engine import VideoSearchEngine
    from quadrag.generation.rag_generator import RAGGenerator
    from quadrag.video.domain_manager import ensure_domain_view
    from _common import (
        extract_letter, format_mc_prompt, search_image_index_by_text,
    )

    yt = video["youtube_id"]
    lens = lens_for_video(video, slice_qa)
    view_name = ensure_domain_view(yt, lens)
    engine = VideoSearchEngine(yt, domain_view_name=view_name)
    generator = RAGGenerator()

    with patched_weights(**NO_LENS_WEIGHTS):
        fusion = ResultFusion()
        per_index = {
            IndexType.AUDIO: engine.search_audio_index(qa["question"], top_k=3),
            IndexType.IMAGE: search_image_index_by_text(engine.video_info, qa["question"], top_k=3),
            IndexType.DESCRIPTION: engine.search_description_index(qa["question"], top_k=3),
            IndexType.DOMAIN: [],
        }
        fused = fusion.fuse_results(per_index, top_k=10)
        prompt = format_mc_prompt(qa)
        resp = with_retries(lambda: generator.generate_answer(prompt, fused), tries=3)
        predicted = extract_letter(resp.answer, qa["options"])
        return {
            "predicted": predicted,
            "answer": resp.answer,
            "grounded": resp.grounded,
            "citations": [
                {"timestamp": c.timestamp, "similarity": c.similarity,
                 "source": c.source.value if hasattr(c.source, "value") else str(c.source),
                 "content": c.content[:200]}
                for c in resp.citations
            ],
            "fused_top_sources": [
                r.source.value if hasattr(r.source, "value") else str(r.source)
                for r in fused[:3]
            ],
            "weights": NO_LENS_WEIGHTS,
        }


def row_to_case(row: dict, slice_qa: List[dict]) -> dict:
    """Pick up the original QA definition and format the case record."""
    qa = next((q for q in slice_qa if q["question_id"] == row["question_id"]), None)
    if qa is None:
        return {"error": f"QA {row['question_id']} not in slice"}
    return {
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "youtube_id": row["youtube_id"],
        "task_type": row.get("task_type", ""),
        "sub_category": row.get("sub_category", ""),
        "question": qa["question"],
        "options": qa["options"],
        "gold_letter": qa["answer"],
        "predicted_letter": row.get("predicted"),
        "answer": row.get("answer_raw", ""),
        "grounded": row.get("grounded"),
        "fused_top_sources": row.get("fused_top_sources", []),
        "retrieved_counts": row.get("retrieved_counts", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relax", action="store_true",
                        help="Accept weaker filters if an archetype has no clean candidate.")
    args = parser.parse_args()

    load_backend_env()
    import os
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    slice_data = load_slice()
    rows = load_eval_rows()
    idx = index_by_qa_config(rows)

    print(f"Loaded {len(rows)} QA-result rows across "
          f"{len({r['config'] for r in rows})} configs")

    # --- archetype 1 ---
    a1 = pick_archetype_1(rows, idx)
    print("\nArchetype 1 (dialogue-heavy, deciding=Audio):")
    if a1 is None and args.relax:
        a1 = next((r for r in rows if r["config"] == "audio_only" and r.get("correct")), None)
    print(f"  picked: {a1['question_id'] if a1 else 'NONE'}")

    # --- archetype 2 ---
    a2 = pick_archetype_2(rows, idx)
    print("\nArchetype 2 (purely visual, deciding=Description/Image):")
    if a2 is None and args.relax:
        a2 = next((r for r in rows if r["config"] in {"image_only", "plus_desc"} and r.get("correct")), None)
    print(f"  picked: {a2['question_id'] if a2 else 'NONE'}")

    # --- archetype 3 ---
    print("\nArchetype 3 (lens-dependent, deciding=Domain):")
    candidates_3 = pick_archetype_3_candidates(rows, idx)
    a3_row = None
    a3_no_lens = None
    if candidates_3:
        import pixeltable as pxt
        pxt.init()
        videos_by_id = {v["video_id"]: v for v in slice_data["videos"]}
        # Try each candidate until one's "no lens" rerun is wrong (the
        # signature we need to prove the lens index made the difference).
        for cand in candidates_3:
            qa = next((q for q in slice_data["qa"] if q["question_id"] == cand["question_id"]), None)
            video = videos_by_id.get(cand["video_id"])
            if qa is None or video is None:
                continue
            try:
                no_lens = rerun_no_lens(qa, video, slice_data["qa"])
            except Exception as e:
                print(f"  rerun failed for {cand['question_id']}: {e}")
                continue
            if no_lens["predicted"] != qa["answer"]:
                a3_row = cand
                a3_no_lens = no_lens
                break
            else:
                print(f"  no-lens rerun was still correct for {cand['question_id']}; trying next")
        if a3_row is None and args.relax and candidates_3:
            # Accept a case where full was correct even if no-lens also was.
            a3_row = candidates_3[0]
            print(f"  relaxed pick: {a3_row['question_id']} (lens delta not confirmed)")
    print(f"  picked: {a3_row['question_id'] if a3_row else 'NONE'}")

    # Compose the output.
    payload = {
        "selection_seed": "deterministic-from-eval-runs",
        "archetypes": {
            "1_dialogue_heavy": {
                "deciding_index": "audio",
                "case": row_to_case(a1, slice_data["qa"]) if a1 else None,
                "baseline_config": "image_only",
                "baseline_row": (row_to_case(idx[(a1["question_id"], "image_only")], slice_data["qa"])
                                 if a1 and (a1["question_id"], "image_only") in idx else None),
            },
            "2_purely_visual": {
                "deciding_index": "description_or_image",
                "case": row_to_case(a2, slice_data["qa"]) if a2 else None,
                "baseline_config": "audio_only",
                "baseline_row": (row_to_case(idx[(a2["question_id"], "audio_only")], slice_data["qa"])
                                 if a2 and (a2["question_id"], "audio_only") in idx else None),
            },
            "3_lens_dependent": {
                "deciding_index": "domain",
                "case": row_to_case(a3_row, slice_data["qa"]) if a3_row else None,
                "baseline_config": "full_minus_lens",
                "baseline_rerun": a3_no_lens,
            },
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
