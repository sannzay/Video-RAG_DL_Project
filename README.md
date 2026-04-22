# 🎬 QuadRAG · Four-Index Multimodal RAG for Video

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-ff4b4b.svg)](https://streamlit.io/)
[![Pixeltable](https://img.shields.io/badge/Pixeltable-0.5+-4f46e5.svg)](https://github.com/pixeltable/pixeltable)
[![Tests](https://img.shields.io/badge/tests-117%20passing-16a34a.svg)](#testing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A training-free, semantically-rich video-QA system.** Upload an MP4; QuadRAG
builds four parallel embedding indexes (visual frames, spoken audio, AI frame
descriptions, optional domain-specific lens), then lets you chat with the
video. Every answer is anchored to specific timestamps, and the UI tells you
when the model is just guessing vs. actually grounded in retrieved content.

🚀 **Live apps**

* Backend API → <https://quadrag-backend-production.up.railway.app>
* Frontend UI → <https://video-ragdlproject-hezrklx8rnhbjwucqvegxq.streamlit.app>

---

## What's different about this one

* **Four indexes, all genuinely semantic.** Earlier revisions shipped two
  embedding indexes + a `difflib`-scored text shim. Now every index — image,
  audio, description, *and* domain — is a real vector search over transformer
  embeddings. You can tell the domain index works because it contributes
  citations alongside the description index for queries where the user's
  lens matters ("from a marketing angle, what's the hook?").
* **Lazy, per-context domain views.** Each video can hold up to 5 simultaneous
  domain lenses ("emotions", "marketing", "storytelling" …), built on the first
  `/chat` that needs them and LRU-evicted beyond that. Ask the same video the
  same kind of question in a new lens without reprocessing.
* **Citation grounding.** After the LLM writes an answer, a regex pulls
  `[M:SS]` timestamps it cites and the backend filters the returned
  citations to those within a ±3 s window. If the model's answer has no
  timestamps, the UI flags it as "Ungrounded" — the model may still be
  right, but you know it's not anchored to specific moments.
* **Interactive chat-style onboarding.** New-video setup happens inside the
  chat, not a modal. The bot asks for the file, offers 6 preset domain
  lenses (plus custom input and skip), confirms, runs, and hands off to
  the real conversation — all without leaving the main area.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit frontend                       │
│   In-chat wizard · per-video chat · status dot · previews    │
└──────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       FastAPI backend                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               Background indexing pipeline             │  │
│  │  transcode → frames view → CLIP embed                  │  │
│  │          │            └→ audio chunks → Whisper        │  │
│  │          │                              → text embed   │  │
│  │          └→ per-frame descriptions → text embed        │  │
│  │          └→ per-frame domain captions (lazy, per lens) │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               Retrieval + generation                   │  │
│  │  ./similarity() on each index  →  weighted fusion      │  │
│  │                   → LLM answer + [M:SS] grounding      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Pixeltable (PostgreSQL + pgvector) · per-video table+views  │
│  frames view ┬ description + embed                           │
│              └ domain view 1, 2, …, N · each with own embed  │
└──────────────────────────────────────────────────────────────┘
```

## Models

| Use | Model |
|---|---|
| Chat answer generation | **Meta Llama 3.3 70B Instruct** |
| Frame vision captions (description + domain indexes) | **Google Gemini 2.0 Flash** |
| Audio transcription (audio index) | OpenAI **Whisper-1** |
| Text embeddings (audio / description / domain indexes) | OpenAI **text-embedding-3-small** |
| Visual frame embeddings (image index) | **CLIP** · `openai/clip-vit-base-patch32` (local PyTorch) |

## Tech stack

| Layer | Choice |
|---|---|
| Backend framework | FastAPI + Uvicorn, Python 3.11 |
| Frontend | Streamlit (+ `st.chat_message`, `st.chat_input`, `@st.fragment`) |
| Vector DB | [Pixeltable](https://github.com/pixeltable/pixeltable) 0.5+ (PostgreSQL + pgvector under the hood) |
| Local ML | PyTorch 2.1 CPU, transformers, sentence-transformers |
| Video processing | FFmpeg (transcode, frame extraction, audio extraction) |
| Deployment | Railway (Nixpacks) for the API, Streamlit Cloud for the UI |
| Testing | pytest (unit, ~2 s) + pytest-recording (opt-in VCR integration) |

## Features

### Four semantic indexes
* **Image** — raw video frames, CLIP embeddings, visual similarity search
* **Audio** — Whisper transcripts of ~10 s audio chunks, text-embedding similarity
* **Description** — AI-generated frame captions (what's in each frame), text-embedding similarity
* **Domain** — frame captions rewritten through a user-supplied lens ("emotions", "marketing", …), text-embedding similarity. **Lazy-built on first chat** with a new lens; LRU-evicted at 5 views per video

### Retrieval
* Per-index top-K search with configurable weights (audio 0.30, description 0.25, domain 0.25, image 0.20)
* Min-max score normalization, weighted fusion, 2-second timestamp deduplication
* Returns top-10 fused results by default

### Generation + grounding
* LLM prompt asks for citations in `[M:SS]` format
* Post-processing extracts cited timestamps and filters citations to matching chunks within ±3 s
* `ChatResponse.grounded: bool` flag surfaces how trustworthy the answer is

### Operations
* **Thread-safe processing state store** with atomic disk snapshots; status survives restarts
* **Adaptive frame sampling** — 40/60/80/100 frames for <5 min / <30 min / <1 h / <2 h videos, 100–150 for longer
* **Auto-polling UI** — `@st.fragment(run_every=5)` updates the processing view without a full rerun
* **Per-browser video preview** — the uploaded file renders inside the chat during setup and in a collapsible header expander during chat

## Prerequisites

* Python 3.11 (pinned in `backend/runtime.txt`)
* FFmpeg on PATH (for local dev; Railway's Nixpacks ships it)
* API keys (see [Configuration](#configuration))
* ≥ 4 GB free disk for the dev venv (PyTorch + transformers are chonky)

## Quick start

### Local dev

```bash
git clone https://github.com/sannzay/Video-RAG_DL_Project.git
cd Video-RAG_DL_Project

# backend venv + deps
./setup_env.sh

# set API keys
cp backend/.env.example backend/.env   # or create one; see Configuration
$EDITOR backend/.env

# start the API (terminal 1)
./start_backend.sh                     # http://localhost:8000

# start the UI (terminal 2)
./start_frontend.sh                    # http://localhost:8501
```

The frontend defaults to `QUADRAG_API_URL=http://localhost:8000`. To point
at a remote backend instead:

```bash
export QUADRAG_API_URL=https://your-backend.example.com
./start_frontend.sh
```

### Deploy

* **Backend → Railway.** `railway up` from the repo root. Needs
  `OPENAI_API_KEY` and `OPENROUTER_API_KEY` set in the project's variables.
  See `railway.toml` and `nixpacks.toml` for the build recipe — do not
  reorder the pip installs in `nixpacks.toml` without testing; `numpy<2` +
  CPU-only torch order matters for image size.
* **Frontend → Streamlit Cloud.** Point at `frontend/app.py` on the `main`
  branch. Set `QUADRAG_API_URL` in the app's Secrets to the Railway URL.

## Usage

1. Click **➕ New video** in the sidebar.
2. Upload an MP4 in the chat bubble that appears.
3. Pick a domain lens from the presets (😊 Emotions, 🎬 Actions,
   📢 Marketing, 📚 Educational, 🎭 Storytelling, 🏞️ Travel), type your own,
   or skip.
4. Confirm. Indexing runs with an auto-polling progress bar; chat input is
   disabled until eager indexes finish.
5. When the bot says "✅ Indexing complete", ask anything. Answers
   come back with timestamped citations; click **Citations** to see the
   retrieved chunks.

## Configuration

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Whisper transcription + text embeddings |
| `OPENROUTER_API_KEY` | ✅ | Chat + vision model provider |
| `QUADRAG_API_URL` | frontend | Override the backend base URL (default `http://localhost:8000`) |
| `QUADRAG_CHAT_TIMEOUT_SEC` | frontend | Chat HTTP timeout, default 120 |
| `QUADRAG_UPLOAD_TIMEOUT_SEC` | frontend | Upload HTTP timeout, default 60 |
| `QUADRAG_STATUS_POLL_TIMEOUT_SEC` | frontend | `/status` HTTP timeout, default 10 |
| `QUADRAG_STATUS_POLL_INTERVAL_SEC` | frontend | Min seconds between `/status` polls, default 2 |
| `PORT` | backend | Railway sets this automatically |
| `RAILWAY_ENVIRONMENT` | backend | Auto-set by Railway; flips path resolution to `/app/data/…` |

### Tuning knobs

All in `backend/src/quadrag/config.py`; override via environment variables of the same name.

| Setting | Default | Notes |
|---|---|---|
| `SPLIT_FRAMES_COUNT` | 40 | Fallback when duration can't be read; `calculate_frame_count` overrides per-duration (40 → 60 → 80 → 100 → 100-150 for ≤5m / ≤30m / ≤1h / ≤2h / longer) |
| `AUDIO_CHUNK_LENGTH` | 10 s | Whisper chunk size |
| `AUDIO_OVERLAP_SECONDS` | 1 s | Chunk overlap |
| `TOP_K_{AUDIO,IMAGE,DESCRIPTION,DOMAIN}` | 3 | Per-index top-K before fusion |
| `FUSION_TOP_K` | 10 | Citations returned after fusion |
| `WEIGHT_{AUDIO,IMAGE,DESCRIPTION,DOMAIN}` | 0.30 / 0.20 / 0.25 / 0.25 | Fusion weights |
| `FUSION_DEDUP_WINDOW_SEC` | 2.0 | Collapse near-duplicate timestamps |
| `MAX_DOMAIN_VIEWS_PER_VIDEO` | 5 | LRU cap on concurrent domain lenses per video |
| `CITATION_TIMESTAMP_TOLERANCE_SEC` | 3.0 | Grounding window (±) for [M:SS] → retrieved chunk matching |

## Project structure

```
Video-RAG_DL_Project/
├── backend/
│   ├── api.py                          # FastAPI entrypoint; lifespan + endpoints
│   ├── requirements.txt                # Production deps
│   ├── requirements-dev.txt            # pytest + pytest-recording + pytest-asyncio
│   ├── runtime.txt                     # Python 3.11.9 pin for Railway
│   └── src/quadrag/
│       ├── config.py                   # Pydantic Settings
│       ├── models.py                   # API request/response shapes
│       ├── utils.py                    # FFmpeg, frame-count schedule, monitoring
│       ├── state/                      # ProcessingStateStore (thread-safe, disk-backed)
│       ├── video/
│       │   ├── processor.py            # Ingestion + orphan cleanup
│       │   ├── indexer.py              # All four indexes live here
│       │   ├── functions.py            # Pixeltable UDFs (CLIP resize, vision caption, …)
│       │   ├── domain_manager.py       # Lazy domain-view creation + LRU
│       │   └── registry.py             # Per-video metadata, JSON-backed
│       ├── retrieval/
│       │   ├── search_engine.py        # Per-index .similarity() queries
│       │   └── fusion.py               # Normalize, weight, dedup, top-K
│       └── generation/
│           └── rag_generator.py        # Chat answer + citation grounding
├── frontend/
│   ├── app.py                          # Streamlit UI (in-chat wizard, auto-poll, per-video chat)
│   └── requirements.txt
├── tests/
│   ├── unit/                           # 117 tests, ~2 s, no external services
│   ├── integration/                    # Opt-in via `pytest -m integration`, VCR-replayed
│   ├── fixtures/sample.mp4             # 2.9 MB Kandima Maldives clip
│   └── README.md                       # VCR record/replay workflow
├── scripts/
│   └── benchmark_indexing.py           # Time each index build for a sample video
├── .streamlit/config.toml              # Streamlit theme (indigo on slate)
├── railway.toml                        # Railway service config
├── nixpacks.toml                       # Nixpacks build recipe
├── pytest.ini                          # Default-excludes integration tests
├── CHANGELOG.md                        # Per-step refactor notes
├── CLAUDE.md                           # Onboarding doc for contributors
└── README.md                           # This file
```

## Testing

```bash
# unit only (default)
backend/.venv/bin/python -m pytest

# integration (needs VCR cassettes or live API keys; see tests/README.md)
pytest tests/integration -m integration --record-mode=once  # first-time record
pytest tests/integration -m integration                     # replay
```

117 unit tests cover: result fusion, citation grounding, frame-count
bucket boundaries, the Whisper-JSON text extractor, `ProcessingStateStore`
concurrency + persistence, domain-view registry (add/touch/drop/LRU),
registry round-trip schema migration (three generations), and
`VideoIndexInfo` legacy-dict tolerance.

## Performance expectations

Indexing times for a ~20 s clip (Kandima Maldives 2.9 MB fixture):

| Stage | Approx time |
|---|---|
| Transcode + frame view | 5–10 s |
| Audio index (Whisper + embed) | 5–10 s |
| Image index (CLIP) | 10–20 s |
| Description index (vision + embed) | 20–30 s |
| **Eager total (what you wait for before chatting)** | **~40–70 s** |
| Domain index (lazy, on first chat with a new lens) | +20–40 s |

Subsequent chats with the same video + same lens: ~1–3 s each. Fresh lens
triggers another domain-view build (~20–40 s) then caches.

Processing time scales roughly linearly with duration (frame count
schedule is 40/60/80/100 for <5 min / <30 min / <1 h / <2 h).

## Limitations

* MP4 (H.264) only. H.264 High profile is transcoded to Main on ingest;
  other codecs aren't tested.
* 500 MB / 2-hour upload cap enforced in `validate_video_size`.
* English-primary Whisper defaults; multilingual works but hasn't been
  tuned.
* Per-video lens cap is 5 (LRU). Switching between more than 5 lenses on
  the same video triggers rebuilds.
* First chat with a brand-new lens on a long video can exceed the 60 s
  edge-proxy cap on Railway. UI tells you to retry; the second request
  hits cache and returns fast.

## Troubleshooting

**Video says "Processing…" forever.** Auto-poll hits `/status` every 5 s;
if it's genuinely stuck, check `railway logs` and look for
`Failed to create <Index> Index: …`. Most often it's an API auth issue
or a Pixeltable migration needed by a version bump.

**`/chat` returns 200 but chat UI shows "Ungrounded".** That's expected
when the LLM's answer doesn't contain `[M:SS]` tokens. The citations are
the raw retrieved chunks; the badge just flags that they're not tied to
specific model-cited moments.

**Frontend shows "Connection issue" briefly after first load.** The
health probe only flips red after two consecutive failed probes, so a
single slow cold-start shouldn't trigger it. If it persists, double-check
`QUADRAG_API_URL` in the frontend's Secrets.

**Chat input is disabled and I'm not in the wizard.** Something upstream
reset flow state. Click **✕ Cancel setup** in the sidebar; it clears the
wizard and the input re-enables.

## Contributing

1. Fork + feature branch
2. `pip install -r backend/requirements.txt -r backend/requirements-dev.txt`
3. Make your changes. Add tests.
4. `pytest` locally; ensure 117 pass (or more, if you added)
5. Open a PR

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

* [Pixeltable](https://github.com/pixeltable/pixeltable) for the multimodal
  table abstraction that underpins every index.
* [Streamlit](https://streamlit.io/) for the chat primitives +
  `@st.fragment` that make the in-chat wizard possible.
* [Railway](https://railway.app/) for Nixpacks + the simple deploy story.

---

**Questions / issues** → [GitHub Issues](https://github.com/sannzay/Video-RAG_DL_Project/issues) ·
**Email** → [sannzayreddy@gmail.com](mailto:sannzayreddy@gmail.com)
