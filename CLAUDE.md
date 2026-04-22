# CLAUDE.md — Video-RAG_DL_Project (QuadRAG)

> Snapshot of the repo as of 2026-04-21, after the 13-step incremental
> refactor described in `CHANGELOG.md`. Goal of this file: give Claude (and
> any new contributor) enough context to pick up the project cold and reason
> about the current code, the wiring between layers, and the remaining
> known-unknowns.

---

## 1. What this project is

**QuadRAG** — a training-free multimodal video-QA system. Upload an MP4, the
backend builds four parallel semantic indexes over it, and a chat UI answers
questions with cited timestamps.

The "four indexes" are the central idea. **All four are now genuinely
semantic** (post-Step-7/8 — the original code shipped two of them as fuzzy
string matching):

| # | Index        | Signal                                  | Embedding                         | Search path                   |
|---|--------------|-----------------------------------------|-----------------------------------|-------------------------------|
| 1 | **Image**    | Raw sampled frames                      | CLIP `clip-vit-base-patch32`      | `.similarity(query_image)`    |
| 2 | **Audio**    | Whisper transcripts of 10 s audio chunks| OpenAI `text-embedding-3-small`   | `.similarity(query_text)` on `transcript_text` |
| 3 | **Description** | GPT-4o-mini caption per frame        | OpenAI `text-embedding-3-small`   | `.similarity(query_text)` on `description` |
| 4 | **Domain**   | GPT-4o-mini caption per frame with user-supplied domain context baked into the prompt | OpenAI `text-embedding-3-small` | `.similarity(query_text)` on `domain_caption`, per-context Pixeltable view |

Results are score-normalized, weighted, deduplicated, and fused. The top-k
chunks are passed as context to **Groq `llama-3.3-70b-versatile`**, which
writes the answer. A post-processing pass (`apply_citation_grounding`)
filters the returned citations to those the LLM actually cited via
`[M:SS]` timestamps in its answer, and sets a `grounded` flag the UI
surfaces as a badge when False.

---

## 2. Tech stack (actual, from the code)

- **Backend:** FastAPI 0.115+, Uvicorn, Python 3.11.9 (pinned in `backend/runtime.txt`)
- **Frontend:** Streamlit 1.40+ (single `frontend/app.py`)
- **Vector DB:** [Pixeltable](https://github.com/pixeltable/pixeltable) 0.5.28 — manages Postgres + pgvector under the hood
- **Local ML:** PyTorch 2.1.x CPU-only, `transformers`, `sentence-transformers`, CLIP (via Pixeltable)
- **Hosted APIs:** OpenAI (Whisper, GPT-4o-mini *via Pixeltable's async* `pxt_openai.vision`, `text-embedding-3-small`) + Groq (chat completions)
- **Media:** FFmpeg (`subprocess`), moviepy, Pillow
- **Deployment:** Railway + Nixpacks (`railway.toml`, `nixpacks.toml`, `Procfile`)
- **Tests:** pytest 8, pytest-recording (VCR), pytest-asyncio

`google-generativeai` is **gone** — it was installed but never meaningfully used.

---

## 3. Repo layout

```
Video-RAG_DL_Project/
├── backend/
│   ├── api.py                      # FastAPI entrypoint, holds endpoints + lifespan
│   ├── requirements.txt            # Production deps; numpy<2, torch 2.1.x pinned
│   ├── requirements-dev.txt        # pytest + pytest-recording + pytest-asyncio
│   ├── pyproject.toml              # quadrag package def (hatchling)
│   ├── runtime.txt                 # python-3.11.9
│   └── src/quadrag/
│       ├── config.py               # Pydantic Settings (env-backed, 30+ settings)
│       ├── models.py               # Pydantic request/response, including ChatResponse.grounded
│       ├── utils.py                # FFmpeg wrappers, frame-count math, monitoring
│       ├── state/                  # Step 4-5: thread-safe processing-state store
│       │   └── processing_state.py # ProcessingStateStore — lock + disk snapshot + cleanup hook
│       ├── video/
│       │   ├── processor.py        # VideoProcessor + cleanup_partial_pixeltable_artifacts()
│       │   ├── indexer.py          # VideoIndexer — builds all four indexes
│       │   ├── functions.py        # Pixeltable UDFs (resize_image, extract_text_from_chunk)
│       │   ├── registry.py         # VideoIndexInfo + domain_views map + JSON persistence + RLock
│       │   └── domain_manager.py   # Step 8: ensure_domain_view() with LRU eviction
│       ├── retrieval/
│       │   ├── search_engine.py    # VideoSearchEngine — per-index semantic search
│       │   └── fusion.py           # ResultFusion — normalize/weight/dedup
│       └── generation/
│           └── rag_generator.py    # Groq call + apply_citation_grounding() post-process
├── frontend/
│   ├── app.py                      # Streamlit UI with debounced polling, grounded badge
│   └── requirements.txt            # Version-pinned deps
├── tests/
│   ├── conftest.py                 # Shared sys.path setup
│   ├── fixtures/                   # sample.mp4 lives here (gitignored)
│   ├── integration/                # VCR-backed end-to-end tests (opt-in)
│   │   ├── conftest.py             # Pixeltable isolation, cassette guard, sample-video skip
│   │   ├── cassettes/              # VCR recordings (commit once made)
│   │   ├── test_chat_endpoint.py
│   │   └── test_indexing_pipeline.py
│   ├── unit/                       # 116 tests, runs in ~2 s
│   │   ├── test_citation_grounding.py
│   │   ├── test_domain_view_registry.py
│   │   ├── test_extract_text_from_chunk.py
│   │   ├── test_frame_count.py
│   │   ├── test_fusion.py
│   │   ├── test_processing_state.py
│   │   ├── test_registry_round_trip.py
│   │   └── test_state_persistence.py
│   └── README.md                   # Record/replay VCR workflow
├── scripts/
│   └── benchmark_indexing.py       # Time each indexing stage (Step 9)
├── data/
│   ├── videos/                     # Uploaded MP4s (gitignored)
│   └── cache/                      # Pixeltable cache + video_registry.json + processing_state.json
├── pytest.ini                      # Default excludes `-m integration`
├── nixpacks.toml                   # Railway build recipe
├── railway.toml                    # Railway runtime config
├── Procfile                        # `cd backend && python3.11 api.py`
├── start.sh / start_backend.sh / start_frontend.sh / stop_servers.sh
├── setup_env.sh
├── README.md
├── CHANGELOG.md                    # Per-step writeup of the 13-step refactor
└── CLAUDE.md                       # This file
```

### Invariants worth remembering

- `backend/src/quadrag/` is the installable package; `backend/api.py` lives outside it and inserts `backend/src` onto `sys.path` at the top of the file.
- Heavy modules (`pixeltable`, `VideoIndexer`, `VideoSearchEngine`, `RAGGenerator`) are **lazy-loaded inside endpoints**, not at import time. The opening ~90 lines of `api.py` exist to avoid a numpy-source-directory import collision and to set the asyncio policy to `DefaultEventLoopPolicy` before anything pulls in `uvloop`.
- The `store = ProcessingStateStore(...)` object and the `_VIDEO_REGISTRY` dict are the two pieces of shared module-level state. Both are now lock-protected and disk-backed.

---

## 4. Data flow

### Upload → index
1. `POST /upload-video` — writes MP4 to `data/videos/<uuid>.mp4`, runs `xattr -c` (macOS quirk), calls `validate_video_size` (≤500 MB, ≤2 h), then fires `asyncio.create_task(_process_video_background(...))` and returns the video_id.
2. `_process_video_background` transcodes to H.264 Main + AAC (Pixeltable rejects H.264 High profile), then spawns a daemon `threading.Thread` running `_process_video_sync`, which creates a fresh event loop and runs `_process_video_async`. The thread isolation exists because Pixeltable's `nest_asyncio` cannot patch uvloop — FastAPI's default loop trips it.
3. `_process_video_async` calls in sequence: `VideoProcessor.process_video` → `create_image_index` → `create_audio_index` → `create_description_index` → (if `domain_context` supplied) `create_domain_index`.
4. State lives in `store` (a `ProcessingStateStore`) plus the JSON registry at `data/cache/video_registry.json`. `store` also snapshots to `data/cache/processing_state.json` on every mutation and is loaded back on startup via the FastAPI `lifespan` hook.

### Query → answer
1. `POST /chat` with `{query, video_id, domain_context?, session_id?}`.
2. If `domain_context` is set: `ensure_domain_view(video_id, domain_context)` looks up the per-context view by normalized-context hash; if absent, LRU-evicts one of the existing ≤5 views and synchronously builds a new one via `VideoIndexer.create_domain_index`. Cache hit just bumps `last_accessed`.
3. `VideoSearchEngine` (constructed with the resolved `domain_view_name`) runs `search_all_indexes` — each of the four `search_*_index` methods issues a `.similarity()` query on the relevant embedding column.
4. `ResultFusion.fuse_results` min-max normalizes, applies weights (audio 0.30, image 0.20, desc 0.25, domain 0.25), dedupes within `settings.FUSION_DEDUP_WINDOW_SEC` (=2 s), sorts, and returns top `FUSION_TOP_K` (=10).
5. `RAGGenerator.generate_answer` builds a grouped-by-source prompt (asking the model to cite timestamps in `[M:SS]` form) and calls Groq with `settings.GROQ_TEMPERATURE` / `settings.GROQ_MAX_TOKENS`.
6. `apply_citation_grounding` parses `[M:SS]` references out of the answer, filters `retrieved_results` to citations within `settings.CITATION_TIMESTAMP_TOLERANCE_SEC` (=3 s) of a cited timestamp, and sets `grounded` accordingly. The UI renders an "Ungrounded response" caption when `grounded=False`.

### Adaptive frame sampling
`utils.calculate_frame_count(duration_sec)`:
- <5 min → 45 frames, <30 min → 90, <1 h → 120, <2 h → 180, else up to 300.

### Audio indexing (post-Step-6)
`create_audio_index` now explicitly `.collect()`s all Whisper transcriptions during indexing, then builds an `add_embedding_index` on `transcript_text`. The keyword-overlap fallback in `search_audio_index` is retained but wrapped with a warning log — it should never fire under normal operation; if it does, indexing didn't finish.

### Domain indexing (post-Step-7/8)
`create_domain_index(video_id, domain_context)` creates a view named `{video_table_name}_domain_{blake2b(context, 4)}` (an 8-char hex hash of the lowercased/stripped context), adds a `domain_caption` computed column driven by `pxt_openai.vision(prompt, resized_frame, ...)` with the domain context baked into the prompt literal, pre-computes captions via `.collect()`, and builds a text-embedding index on `domain_caption`. `add_domain_view(...)` registers it under the context hash. Up to `settings.MAX_DOMAIN_VIEWS_PER_VIDEO` (=5) coexist per video; the 6th triggers LRU eviction of the least-recently-touched view (both registry entry + Pixeltable view dropped).

### Concurrency (post-Step-9)
All per-frame OpenAI calls (description + domain) use `pxt_openai.vision`, which is async and uses OpenAI's response headers to throttle adaptively. No custom rate-limiting loop, no `time.sleep(2)`, no manual `OpenAI()` client construction inside `VideoIndexer`.

---

## 5. Configuration & deployment

### Settings (`backend/src/quadrag/config.py`)
Loaded from `.env` via `pydantic-settings`. Notable groups:

- **API Keys** — `GROQ_API_KEY`, `OPENAI_API_KEY`. `GOOGLE_API_KEY` still accepted (unused; kept only so existing `.env` files don't error).
- **Models** — `GROQ_MODEL`, `IMAGE_CAPTION_MODEL`, `TEXT_EMBEDDING_MODEL`, `IMAGE_EMBEDDING_MODEL`, `AUDIO_TRANSCRIPT_MODEL`.
- **Processing** — `SPLIT_FRAMES_COUNT` (default, overridden by adaptive sampling), `AUDIO_CHUNK_LENGTH`, `AUDIO_OVERLAP_SECONDS`.
- **Retrieval** — `TOP_K_*`, `FUSION_TOP_K`, `FUSION_DEDUP_WINDOW_SEC`, `WEIGHT_*`.
- **LLM / Vision** — `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `VISION_TEMPERATURE`, `VISION_MAX_TOKENS`.
- **Domain Index** — `MAX_DOMAIN_VIEWS_PER_VIDEO` (=5). The legacy difflib knobs (`DOMAIN_SIMILARITY_THRESHOLD`, `DOMAIN_SEQUENCE_WEIGHT`, `DOMAIN_WORD_OVERLAP_WEIGHT`) are unreferenced now but kept so old `.env` files don't complain.
- **Citation Grounding** — `CITATION_TIMESTAMP_TOLERANCE_SEC` (=3.0).
- **Timeouts (server-side defaults for .env)** — `UPLOAD_TIMEOUT_SEC`, `CHAT_TIMEOUT_SEC`, `STATUS_POLL_TIMEOUT_SEC`.
- **Paths** — `DATA_DIR`, `VIDEO_DIR`, `CACHE_DIR`. Still relative (`"../data"`) for local dev; `get_video_dir()` / `get_cache_dir()` switch to `/app/data/...` when `RAILWAY_ENVIRONMENT` is set. Run the backend from `backend/` (that's what `start_backend.sh` does) so the relative paths resolve correctly.

### Frontend env vars (`frontend/app.py`)
- `QUADRAG_API_URL` — defaults to `http://localhost:8000` (Step 2 flipped this from the author's Railway prod URL).
- `QUADRAG_STATUS_POLL_INTERVAL_SEC` — poll debounce, default 2.0 s.
- `QUADRAG_CHAT_TIMEOUT_SEC` / `QUADRAG_UPLOAD_TIMEOUT_SEC` / `QUADRAG_STATUS_POLL_TIMEOUT_SEC` — override client-side request timeouts.

### Local dev
1. `./setup_env.sh` — creates venvs, installs deps.
2. `./start_backend.sh` — checks `.env`, runs `uvicorn` on :8000.
3. `./start_frontend.sh` — runs Streamlit on :8501. Default URL is now localhost.
4. `./stop_servers.sh` — kills both.
5. Tests: `backend/.venv/bin/python -m pytest` (unit, ~2 s). Add `-m integration` after following the VCR-recording workflow in `tests/README.md`.

### Railway deploy
`nixpacks.toml` is load-bearing — it installs numpy 1.x first, then CPU-only PyTorch, then the rest of `requirements.txt`, to keep the image under Railway's size limits. Don't "clean it up" without a real deploy test. `Procfile` starts only the backend; the frontend is a separate Streamlit Cloud deployment whose `QUADRAG_API_URL` should be set to the Railway backend URL.

---

## 6. Known rough edges (remaining)

Most of the big ones from the pre-refactor snapshot are fixed — see
`CHANGELOG.md` for the per-step rundown. What's still on the to-do list:

### Worth doing soon
- **Relative path `../data`** in `config.DATA_DIR` and `registry._REGISTRY_FILE` — still fragile. Not urgent because both `start_backend.sh` (cd backend) and Railway (`RAILWAY_ENVIRONMENT` → absolute `/app/data`) paper over it, but a direct `python backend/api.py` from the repo root writes snapshots to the parent directory. Fix by plumbing `settings.get_cache_dir()` through every caller.
- **`/videos` endpoint has no pagination.** Returns every video in one list. Fine for dozens, bad for thousands.
- **Legacy Step-7 registry entries** land in `domain_views["legacy"]` — the original `domain_context` is unknown, so the view is reachable by admin tooling only, not by a natural `/chat` call. Eventually ages out via LRU. If you see many `"legacy"` entries in production, consider a one-shot migration script.

### Worth noting, not fixing
- **`google-generativeai` config key still accepted** — only so existing `.env` files don't refuse to load. No callers.
- **Unused `DOMAIN_SIMILARITY_THRESHOLD` / `DOMAIN_SEQUENCE_WEIGHT` / `DOMAIN_WORD_OVERLAP_WEIGHT` settings** — held for the same .env-compat reason. Delete them once you're confident no one is still on the pre-Step-7 code path.
- **`_save_registry_locked` / `_snapshot_to_disk_locked` write JSON synchronously on every mutation.** Fine at the current scale; if someone uploads hundreds of videos and the registry grows, switch to debounced writes.
- **Streaming `/chat` response** — `generate_streaming_answer` exists but Step 11's citation grounding wasn't applied to it. Fix before wiring the frontend to streaming.

### Intentional trade-offs (not bugs)
- **Lazy domain-view creation happens synchronously in `/chat`.** First query with a fresh `domain_context` can take 30 s–2 min. The frontend's `QUADRAG_CHAT_TIMEOUT_SEC` defaults to 120 s; bump it if you see timeouts on long videos. A background-build + "still preparing" response would be more UX-friendly but more code.
- **Per-video LRU cap of 5 domain views.** If users regularly switch between >5 contexts on the same video, the 6th switch triggers a recompute. Raise `MAX_DOMAIN_VIEWS_PER_VIDEO` or evict smarter if this bites.
- **Integration tests require VCR cassettes.** Default `pytest` runs unit-only (`-m "not integration"`). The integration tests skip gracefully without either cassettes or a sample MP4. Opt-in workflow documented in `tests/README.md`.

---

## 7. Testing state

- **Unit**: 116 tests, ~2 s, no network, no API keys needed. Covered: `ResultFusion`, `extract_text_from_chunk`, frame-count math, registry round-trip (including legacy-schema migration), `ProcessingStateStore` (basic + persistence + concurrency stress + cleanup callback), domain-view registry (add/touch/drop/LRU), citation grounding (extraction regex + filter logic + hallucinated-timestamp honesty).
- **Integration**: three tests, opt-in via `pytest -m integration`. Exercise the full indexing pipeline and `/chat` end-to-end. Use `pytest-recording` for VCR replay — the first run with real keys records cassettes; subsequent runs and CI replay from the committed YAML. See `tests/README.md`.
- **No CI workflow file yet** — adding one is a straightforward follow-up once cassettes are recorded.

---

## 8. When working in this repo

- **Before changing indexing or search behavior**: run the end-to-end test (if cassettes exist) or at minimum the unit suite. The remaining classes of bug this codebase is prone to — Pixeltable quirks, async/sync mismatches — are mostly the kind that parse cleanly but surface at first run with a real video.
- **Prefer `logger` over `print`.** `loguru` is configured project-wide. Step 1 converted ~20 `print`s; don't reintroduce them.
- **Touch `nixpacks.toml` only when you can actually redeploy to Railway.** The install order there is load-bearing for image size.
- **Don't commit `.env` or anything in `data/`.** `.gitignore` covers them; double-check before push.
- **When editing `api.py`'s top-of-file import dance**, understand exactly which numpy import used to fail. The sys.path cleanup exists because running `api.py` from a directory containing a `numpy/` subdir (not the repo root, but some deploy layouts) causes numpy to try to import itself from source.
- **When editing the registry schema**, add a migration branch in `VideoIndexInfo.from_dict` — we already carry migrations for pre-Step-7, Step-7, and Step-8 shapes. Adding a Step-N branch is a few lines.
- **When adding new config settings**, give them a default in `config.py` so existing `.env` files don't reject the extra field. Use `pydantic_settings`' `extra="ignore"` (already configured).
- **When the plan mentions a "spike"**, that means "run a small experiment against the real tool before committing to the approach." Skipped spikes in Steps 6 and 7 were documented as trade-offs in the completion notes — not to be emulated silently.

---

## 9. High-value improvement areas (still open)

Ordered roughly by impact × effort. Not a plan — a menu.

1. **Fix the relative `../data` paths** throughout `config.py` and `registry.py`. Low-risk, 30-minute task; makes the codebase resilient to being run from any cwd.
2. **Streaming `/chat`** — wire `generate_streaming_answer` into a streaming FastAPI response + Streamlit's new chat-stream primitive. Feels much faster to users even with the same total latency.
3. **Background domain-view build** — instead of blocking `/chat` on the first query with a new context, return a "preparing this answer, check back in 60s" response and complete in the background. Removes the timeout risk on long videos.
4. **CI workflow** — GitHub Actions that runs `pytest` on every push + `pytest -m integration` on PRs labelled `full-check`. Requires VCR cassettes committed.
5. **Persist registry in Postgres** — the JSON file works but doesn't scale past a few hundred videos and can be corrupted by concurrent writes across processes (the in-process `_REGISTRY_LOCK` protects threads, not processes).
6. **Docker image for the backend** — `nixpacks.toml` works for Railway but doesn't help anyone running elsewhere. A plain Dockerfile opens self-hosted + generic-cloud deploys.
7. **Hardening**: path-traversal checks on uploaded filenames, filename-length limits, per-IP rate limits on `/chat` and `/upload-video`, structured audit logging.
8. **Observability**: OpenTelemetry traces through `/upload-video` → background thread → `/chat` so slow steps are debuggable in prod.

---

*Last updated: 2026-04-21. If this file drifts from the code, fix the file — it's the onboarding contract.*
