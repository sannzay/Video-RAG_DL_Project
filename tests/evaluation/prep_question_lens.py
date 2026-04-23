"""Phase 1b - derive a question-informed analytical lens per video.

The first eval pass used the Video-MME ``sub_category`` field as the lens
source (e.g. ``"Humanity & History"`` -> ``"historical and cultural
analysis"``). That turned out to be too generic: the ``full`` LensRAG
configuration under-performed ``plus_desc``, because the generic lens
produced captions that didn't focus on the specific things the evaluation
questions actually asked about.

Production usage looks different: a user sets a lens based on the kind of
questions they know they're about to ask. This script simulates that by
summarising each video's QA questions into a single lens phrase via the
chat LLM, and writes the result to ``data/question_lenses.json``. The
downstream domain-view rebuild and the ``full``-config eval then use
those lenses.

Prompt is deliberately conservative: the lens must describe the
analytical *angle* (what to attend to), not the answer itself. The LLM
sees only the question stems, never the gold letter or option text.

Usage:
    cd Video-RAG_DL_Project/backend && source .venv/bin/activate
    cd .. && python tests/evaluation/prep_question_lens.py

Idempotent: the output file is rewritten from scratch each run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from _common import REPO_ROOT, load_backend_env, load_slice

OUTPUT_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "question_lenses.json"


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------

LENS_SYSTEM = (
    "You help users configure a per-video vision-caption lens. A lens is "
    "a short analytical phrase (6 to 14 words) that tells the vision system "
    "what to focus on when describing each frame. Given a set of questions "
    "a user plans to ask about a video, your job is to summarise the "
    "analytical angle those questions share into a single lens phrase.\n\n"
    "Hard rules:\n"
    "  * Describe the lens, not the answer. Do not parrot option text.\n"
    "  * Name the concrete entities / attributes / events the questions "
    "    care about (people, objects, counts, actions, scores, brand names, "
    "    on-screen text, colors, timing, outcomes), not just abstract themes.\n"
    "  * If any question asks about visible text, numbers, logos, or scores, "
    "    the lens MUST include a phrase like \"visible text and numbers\" or "
    "    \"on-screen labels and branding\".\n"
    "  * If any question asks about who did what or sequence of events, the "
    "    lens MUST include \"identifying people and their actions\".\n"
    "  * Keep the lens between 6 and 14 words; no trailing punctuation.\n"
    "  * Do not use the words 'video', 'questions', 'user', or 'lens'.\n\n"
    "Good examples:\n"
    "  - emotional cues interpersonal dynamics and visible dialogue labels\n"
    "  - cooking technique ingredients counts and on-screen recipe text\n"
    "  - athletic play strategy scoring events and jersey numbers\n"
    "  - marketing hooks branding sponsor logos and on-screen captions\n"
    "  - identifying people their actions clothing colors and visible signage\n"
    "  - historical artifacts cultural symbols and on-screen date or place labels"
)


def user_prompt(questions: List[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(questions))
    return (
        f"A user is about to ask these questions about the video:\n\n"
        f"{numbered}\n\n"
        f"Write one short lens phrase that covers the analytical angle "
        f"these questions share. Output only the phrase, nothing else."
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _clean_lens(raw: str) -> str:
    """Normalize the LLM's lens output: single line, no leading bullet, lower case."""
    text = raw.strip()
    # Drop a leading bullet or quote.
    for bad_prefix in ("- ", "* ", "\u2022 ", '"', "'"):
        if text.startswith(bad_prefix):
            text = text[len(bad_prefix):].strip()
    # Drop trailing punctuation.
    text = text.rstrip(".!?\"' ")
    # Single line.
    text = text.split("\n", 1)[0].strip()
    return text


def main() -> int:
    load_backend_env()
    for key in ("OPENROUTER_API_KEY",):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    slice_data = load_slice()

    # Defer heavy imports so the env vars are live first.
    from quadrag.config import get_settings
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
    )

    # Group QA questions by video.
    questions_by_video: Dict[str, List[str]] = {}
    for qa in slice_data["qa"]:
        questions_by_video.setdefault(qa["video_id"], []).append(qa["question"])

    lenses: Dict[str, dict] = {}

    print(f"Generating lenses for {len(slice_data['videos'])} videos")
    print("-" * 70)

    for v in slice_data["videos"]:
        vid = v["video_id"]
        yt = v["youtube_id"]
        qs = questions_by_video.get(vid, [])
        if not qs:
            print(f"  {yt}: no QA; skipping")
            continue

        try:
            resp = client.chat.completions.create(
                model=settings.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": LENS_SYSTEM},
                    {"role": "user", "content": user_prompt(qs)},
                ],
                # Lower temperature than the answer generator: we want the
                # lens to be stable and focused, not creative.
                temperature=0.2,
                max_tokens=40,
            )
            raw = resp.choices[0].message.content or ""
            lens = _clean_lens(raw)
            if not lens or len(lens.split()) > 18:
                # Fall back to a safe generic if the LLM's answer looks broken.
                print(f"  {yt}: LLM lens rejected ({raw!r}); falling back")
                lens = "general video content analysis"
        except Exception as e:
            print(f"  {yt}: lens generation failed: {e}; falling back")
            lens = "general video content analysis"

        lenses[yt] = {
            "lens": lens,
            "n_questions": len(qs),
            "sub_category": next(
                (qa.get("sub_category", "") for qa in slice_data["qa"]
                 if qa["video_id"] == vid), ""
            ),
            "duration_bucket": v.get("duration_bucket"),
        }
        print(f"  {yt:>12} ({v.get('duration_bucket'):>6}): {lens!r}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(lenses, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Coverage: {len(lenses)} / {len(slice_data['videos'])} videos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
