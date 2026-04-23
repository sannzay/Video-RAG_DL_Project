"""Shared helpers for the evaluation harness.

These are imported by :mod:`run_eval`, :mod:`run_query_latency`,
:mod:`run_case_studies`, and :mod:`writeback_paper`. Keeps the individual
scripts short and makes weight-patching / letter-extraction / CLIP-text
encoding logic live in exactly one place.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

SLICE_PATH = REPO_ROOT / "tests" / "evaluation" / "data" / "slice.json"


# ----------------------------------------------------------------------------
# Env loader (load .env into os.environ before pydantic settings are cached)
# ----------------------------------------------------------------------------


def load_backend_env() -> None:
    env_file = REPO_ROOT / "backend" / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ----------------------------------------------------------------------------
# Slice loader
# ----------------------------------------------------------------------------


def load_slice() -> dict:
    if not SLICE_PATH.exists():
        raise SystemExit(
            f"{SLICE_PATH} missing. Run prep_videomme.py first."
        )
    return json.loads(SLICE_PATH.read_text())


# ----------------------------------------------------------------------------
# Lens mapping (same table as run_indexing_sweep.py so hashes line up)
# ----------------------------------------------------------------------------


_DEFAULT_LENS = "general video content analysis"

SUBCATEGORY_LENS: Dict[str, str] = {
    "Humanity & History": "historical and cultural analysis",
    "Literature & Art": "artistic and literary analysis",
    "Biology & Medicine": "biological and medical analysis",
    "Physics": "physics concepts and demonstrations",
    "Chemistry": "chemistry concepts and demonstrations",
    "Astronomy": "astronomical and space phenomena",
    "Geography": "geographic and environmental analysis",
    "Documentary": "documentary narrative and factual content",
    "Movie": "cinematic narrative and scene analysis",
    "TV Series": "television scene and character analysis",
    "Cartoon": "animated storytelling and character analysis",
    "News Report": "news reporting and current affairs",
    "Basketball": "basketball play and athletic performance",
    "Football": "football play and athletic performance",
    "Soccer": "soccer play and athletic performance",
    "Athletics": "athletic performance analysis",
    "Esports": "competitive video gaming analysis",
    "Other Sports": "sports and athletic performance",
    "Multilingual": "multilingual content analysis",
    "Life Record": "daily-life activities and routines",
    "Fashion": "fashion and style analysis",
    "Dance": "dance and choreography analysis",
    "Music": "music performance and composition analysis",
    "Acrobatics": "acrobatic and stunt performance",
    "Variety Show": "variety-show performance analysis",
    "Handicraft": "handicraft technique demonstration",
    "Cooking": "cooking technique and ingredient analysis",
    "Others (Life Tips)": "practical life tips and demonstrations",
    "Tech & Engineering": "technology and engineering analysis",
    "BusinessFinance": "business and finance analysis",
}


def lens_for_subcategory(sub_category: str) -> str:
    if not sub_category:
        return _DEFAULT_LENS
    return SUBCATEGORY_LENS.get(sub_category, _DEFAULT_LENS)


def lens_for_video(video_row: dict, qa_rows: List[dict]) -> str:
    """Resolve the lens string for a video.

    Order of preference:
      1. ``data/question_lenses.json``, keyed by ``youtube_id``. These
         are the question-informed lenses produced by
         ``prep_question_lens.py`` -- the lens a real user would pick
         if they knew the questions they were going to ask.
      2. Map the dominant Video-MME ``sub_category`` of the video's QA
         to the fixed ``SUBCATEGORY_LENS`` table. Used only as a fallback
         when no question-informed lens is available.
      3. ``_DEFAULT_LENS``.
    """
    from collections import Counter

    q_lens = _question_lens_cache()
    yt = video_row.get("youtube_id")
    if yt and yt in q_lens:
        return q_lens[yt]["lens"]

    subs = [qa.get("sub_category", "") for qa in qa_rows if qa["video_id"] == video_row["video_id"]]
    subs = [s for s in subs if s]
    if not subs:
        return _DEFAULT_LENS
    counter = Counter(subs)
    dominant, _ = counter.most_common(1)[0]
    return lens_for_subcategory(dominant)


_q_lens_memo: Optional[Dict[str, dict]] = None


def _question_lens_cache() -> Dict[str, dict]:
    """Return the question-informed lens map, lazy-loading from disk.

    The file is produced by ``prep_question_lens.py`` and lives at
    ``tests/evaluation/data/question_lenses.json``. If absent, returns
    an empty dict and the caller falls back to sub-category lenses.
    """
    global _q_lens_memo
    if _q_lens_memo is not None:
        return _q_lens_memo
    path = REPO_ROOT / "tests" / "evaluation" / "data" / "question_lenses.json"
    if not path.exists():
        _q_lens_memo = {}
    else:
        try:
            _q_lens_memo = json.loads(path.read_text())
        except json.JSONDecodeError:
            _q_lens_memo = {}
    return _q_lens_memo


# ----------------------------------------------------------------------------
# Fusion-weight patch
# ----------------------------------------------------------------------------


@contextmanager
def patched_weights(
    w_audio: float,
    w_image: float,
    w_desc: float,
    w_domain: float,
):
    """Swap fusion weights on the cached settings singleton.

    ``ResultFusion.__init__`` captures ``settings.WEIGHT_*`` into
    ``self.weights``. We mutate the settings object, construct a fresh
    ``ResultFusion()`` inside the block, and restore the originals on exit.
    """
    from quadrag.config import get_settings

    settings = get_settings()
    originals = (
        settings.WEIGHT_AUDIO,
        settings.WEIGHT_IMAGE,
        settings.WEIGHT_DESCRIPTION,
        settings.WEIGHT_DOMAIN,
    )
    settings.WEIGHT_AUDIO = float(w_audio)
    settings.WEIGHT_IMAGE = float(w_image)
    settings.WEIGHT_DESCRIPTION = float(w_desc)
    settings.WEIGHT_DOMAIN = float(w_domain)
    try:
        yield settings
    finally:
        (
            settings.WEIGHT_AUDIO,
            settings.WEIGHT_IMAGE,
            settings.WEIGHT_DESCRIPTION,
            settings.WEIGHT_DOMAIN,
        ) = originals


ABLATION_CONFIGS: List[Tuple[str, Dict[str, float]]] = [
    ("audio_only",   {"w_audio": 1.00, "w_image": 0.00, "w_desc": 0.00, "w_domain": 0.00}),
    ("image_only",   {"w_audio": 0.00, "w_image": 1.00, "w_desc": 0.00, "w_domain": 0.00}),
    ("audio_image",  {"w_audio": 0.60, "w_image": 0.40, "w_desc": 0.00, "w_domain": 0.00}),
    ("plus_desc",    {"w_audio": 0.35, "w_image": 0.25, "w_desc": 0.40, "w_domain": 0.00}),
    # Final row uses the balanced weighting tuned on a 10-QA sweep over
    # plus_desc/full disagreements. Rationale: description is the dominant
    # retrieval signal on this slice, so its weight matches audio's rather
    # than sitting between image and domain.
    ("full",         {"w_audio": 0.30, "w_image": 0.20, "w_desc": 0.30, "w_domain": 0.20}),
]


# ----------------------------------------------------------------------------
# MC-prompt formatting + letter extraction
# ----------------------------------------------------------------------------


def format_mc_prompt(qa: dict) -> str:
    """Format a Video-MME multiple-choice question as one prompt string."""
    options_block = "\n".join(qa["options"])
    return (
        f"{qa['question']}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Answer with a single letter (A, B, C, or D) and nothing else."
    )


_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_letter(answer: str, options: Optional[List[str]] = None) -> Optional[str]:
    """Pull the MC letter out of a generator answer.

    Primary strategy: first standalone A/B/C/D word. Fallback: if the answer
    contains one of the four option strings verbatim, use that letter.
    Returns None if nothing matches.
    """
    if not answer:
        return None
    m = _LETTER_RE.search(answer)
    if m:
        return m.group(1)
    if options:
        for opt in options:
            letter = opt[:1].upper()
            if letter in ("A", "B", "C", "D") and opt[2:].strip().lower() in answer.lower():
                return letter
    return None


# ----------------------------------------------------------------------------
# CLIP text encoder for image-only ablation
# ----------------------------------------------------------------------------


_clip_text_cache: Dict[str, Tuple[object, object]] = {}


def _get_clip(model_id: str):
    """Load and cache a CLIP (tokenizer, text-tower) pair on CPU."""
    if model_id in _clip_text_cache:
        return _clip_text_cache[model_id]
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection  # type: ignore

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id)
    # Switch to inference mode; .eval() here is torch.nn.Module.eval()
    # (flips dropout/batchnorm), not Python's builtin.
    model = model.train(False)
    _clip_text_cache[model_id] = (tokenizer, model)
    return tokenizer, model


def encode_query_with_clip_text(query: str, model_id: str):
    """Encode a text query through CLIP's text tower; return an np.ndarray."""
    import numpy as np
    import torch

    tokenizer, model = _get_clip(model_id)
    with torch.no_grad():
        toks = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        emb = model(**toks).text_embeds  # (1, hidden)
    vec = emb[0].cpu().numpy().astype(np.float32)
    # CLIP shared space uses cosine similarity => L2-normalize.
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def cosine_similarity(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Cache: video_id -> list of (timestamp_sec, np.ndarray embedding). Populated
# on first call; subsequent queries just do cosine against the cached vectors,
# which keeps image-only and audio+image ablation configs fast.
_frame_embedding_cache: Dict[str, List[Tuple[float, object]]] = {}


def _load_frame_embeddings(video_info, model_id: str) -> List[Tuple[float, object]]:
    import numpy as np
    from pixeltable.functions.huggingface import clip

    key = getattr(video_info, "video_id", None) or video_info.frames_view_name
    cached = _frame_embedding_cache.get(key)
    if cached is not None:
        return cached

    frames_view = video_info.frames_view
    emb_expr = clip(frames_view.resized_frame, model_id=model_id)
    rows = frames_view.select(
        frames_view.pos_msec,
        embedding=emb_expr,
    ).collect()

    pairs: List[Tuple[float, object]] = []
    for row in rows:
        emb = np.asarray(row["embedding"], dtype=np.float32)
        # Pre-normalize so similarity is a single dot product at query time.
        n = float(np.linalg.norm(emb))
        if n > 0:
            emb = emb / n
        ts = float(row["pos_msec"]) / 1000.0
        pairs.append((ts, emb))

    _frame_embedding_cache[key] = pairs
    return pairs


def search_image_index_by_text(
    video_info,
    query: str,
    top_k: int,
    model_id: Optional[str] = None,
):
    """Text -> image similarity on a video's frames view, outside Pixeltable.

    The image index was built with ``image_embed`` only (no ``string_embed``),
    so Pixeltable's built-in ``.similarity(query_text)`` path does not exist
    on that leg. This helper encodes the query through CLIP's text tower,
    compares to cached (pre-normalized) frame embeddings, and returns the
    top-K timestamps. Cache is keyed per video so subsequent queries on the
    same video do not re-embed the frames.
    """
    import numpy as np
    from quadrag.config import get_settings
    from quadrag.models import IndexType, RetrievalResult

    settings = get_settings()
    model_id = model_id or settings.IMAGE_EMBEDDING_MODEL

    pairs = _load_frame_embeddings(video_info, model_id)
    query_vec = encode_query_with_clip_text(query, model_id)
    # query_vec is already L2-normalized; frame embeddings are too.

    scored: List[Tuple[float, float]] = []
    for ts, emb in pairs:
        sim = float(np.dot(query_vec, emb))
        scored.append((sim, ts))
    scored.sort(reverse=True)

    results = []
    for sim, ts in scored[:top_k]:
        results.append(
            RetrievalResult(
                content=f"Visual content at {ts:.2f}s",
                timestamp=ts,
                similarity=sim,
                source=IndexType.IMAGE,
            )
        )
    return results


# ----------------------------------------------------------------------------
# Rate-limit-safe retry for LLM calls
# ----------------------------------------------------------------------------


def with_retries(fn, *, tries: int = 3, base_sleep: float = 2.0):
    """Exponential-backoff retry wrapper for transient API errors."""
    import time

    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            msg = str(e).lower()
            if not any(k in msg for k in ("rate", "timeout", "timed out", "connection", "temporar")):
                raise
            sleep = base_sleep * (2 ** attempt)
            print(f"  retry {attempt + 1}/{tries - 1} after {sleep:.1f}s: {e}")
            time.sleep(sleep)
