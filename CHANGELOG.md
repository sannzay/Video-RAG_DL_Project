# Changelog

All notable changes to this project land here. Versioning is casual — one
release per landed refactor milestone. Dates reflect when the work was done.

## [0.2.0] — 2026-04-21 — "Make the product match its claims"

Thirteen-step incremental refactor of the codebase that the original README
advertised a multimodal-semantic system but the code shipped as two
semantic indexes, one keyword-match fallback, and one `difflib`-scored text
shim. After this release, all four indexes are genuinely semantic, state
is thread-safe and restart-resilient, indexing is ~3–5× faster (estimated;
benchmark tooling ships in Step 9), and the product is covered by 116 unit
tests plus an opt-in integration-test harness.

### Migration notes

* **Frontend `QUADRAG_API_URL`**: pre-0.2, the frontend hardcoded the
  author's Railway backend URL as the default. Post-0.2, the default is
  `http://localhost:8000`. If you were relying on the prod URL, export
  `QUADRAG_API_URL=https://your-backend` before running Streamlit.
* **Registry JSON**: schema migrated twice (pre-Step-7 inline captions →
  Step-7 single `domain_view_name` → Step-8 `domain_views` map keyed by
  normalized-context hash). `VideoIndexInfo.from_dict` silently accepts
  all three shapes, so existing `data/cache/video_registry.json` files
  load cleanly. Step-7 legacy entries land in `domain_views["legacy"]`
  and age out via LRU eviction.
* **Python 3.11** is required (pinned in `backend/runtime.txt`). The
  development venv lives at `backend/.venv`.
* **`google-generativeai`** removed from dependencies. The env var is
  still tolerated but has no callers.

### Step-by-step

Each step is a single self-contained change, landed in the order below. See
the commit for the full diff.

**Step 1 — Dead code, logging, exception hygiene**
Removed `_current_domain_context` module global (the bug that made pre-0.2
domain captions always use the default context), `google-generativeai`,
unused `ThreadPoolExecutor` imports, broken `test_audio_search.py` (parse
error). Converted 18 `print(...)` calls to `logger`. Replaced four bare
`except:` blocks with explicit `except Exception: logger.exception(...)`.
Fixed `VideoIndexInfo.to_dict()` to round-trip `domain_captions` /
`domain_context` — they were silently dropped on every restart.

**Step 2 — Configuration hygiene**
Moved every hardcoded magic number (temperatures, token limits, batch
sizes, sleep durations, fusion dedup window, domain similarity threshold)
into `config.Settings`. Flipped the frontend default URL to localhost.
Pinned upper bounds on `frontend/requirements.txt` so a Streamlit breaking
change can't surprise us.

**Step 3 — Unit-test scaffolding**
Added `backend/requirements-dev.txt` (pytest, pytest-asyncio,
pytest-recording). Created `tests/` with 42 unit tests for `ResultFusion`,
the Whisper-text extractor, `utils.calculate_frame_count` bucket
boundaries, and the Step-1 registry round-trip regression. Deleted three
ad-hoc smoke scripts.

**Step 4 — Thread-safe `ProcessingStateStore` (in-memory)**
Replaced `api.py`'s four module-level dicts with a single
`ProcessingStateStore` instance backed by a `threading.RLock`. Exposed the
11 mutation + read methods actually used by the backend. Migrated ~20
call sites. Added 22 unit tests including a 16-thread × 500-op
concurrency stress test.

**Step 5 — Persistence + registry lock + orphan cleanup**
`ProcessingStateStore` now snapshots to `data/cache/processing_state.json`
atomically (tmp + rename) on every mutation, and loads from disk via a
FastAPI `lifespan` hook. Graceful on corrupt/missing files — returns
empty store + warning, never crashes. Registry (`_VIDEO_REGISTRY`) gained
a module-level `RLock`. `mark_failed` takes an optional
`on_fail_cleanup` callback (wired to
`cleanup_partial_pixeltable_artifacts`) that drops orphaned Pixeltable
views after a mid-processing crash. 11 new tests.

**Step 6 — Audio index: semantic for real**
`create_audio_index` now `.collect()`s every Whisper transcript up-front
and builds an `add_embedding_index` on `transcript_text`. The
keyword-overlap fallback in `search_audio_index` is preserved but wrapped
in a warning log — post-0.2 it should never fire; if it does, the logs
show why. Semantic queries like "what mood is being described" now return
relevant chunks instead of empty.

**Step 7 — Domain index: Pixeltable-native with embeddings**
Deleted ~140 lines of per-batch Python loop calling OpenAI per frame and
writing captions into `video_registry.json`. Deleted ~110 lines of
`difflib.SequenceMatcher` scoring in `search_domain_index`. Replaced with
a proper Pixeltable view + `add_computed_column(domain_caption=...)` +
`add_embedding_index` — semantic search via `.similarity()`. Single view
per video at this step; multi-view is Step 8. Rewrote the
`describe_image_with_domain` UDF to take `domain_context` as an explicit
arg (no more module global). Registry `to_dict`/`from_dict` tightened to
only emit current-schema fields, with legacy-dict tolerance.

**Step 8 — Multi-domain views with LRU eviction**
One video can now have up to `MAX_DOMAIN_VIEWS_PER_VIDEO` (=5) concurrent
domain views keyed by `blake2b(normalized_context, 4)`. New
`domain_manager.ensure_domain_view(video_id, domain_context)` is the
single entry point both the eager upload path and the lazy `/chat` path
go through — cache hit touches `last_accessed`; cache miss evicts LRU if
at capacity and synchronously builds. Registry schema migrated (Step 7's
flat `domain_view_name` carried into `domain_views["legacy"]`). 17 new
tests (registry round-trip rewrite + new domain-view-registry suite).

**Step 9 — Indexing throughput + concurrency**
Swapped the custom synchronous `describe_image` / `describe_image_with_domain`
UDFs for Pixeltable's native async `pxt_openai.vision`, which uses OpenAI
response headers to throttle adaptively. Retired both custom UDFs along
with the `openai.OpenAI()` client instance in `VideoIndexer.__init__`.
Added `scripts/benchmark_indexing.py` that times each of the four
indexing stages end-to-end (requires sample video + live API keys, documented
in the script).

**Step 10 — Integration tests with VCR replay**
Scaffolded `tests/integration/` with three tests exercising the full
indexing pipeline and `/chat` end-to-end via `TestClient`. `pytest-recording`
captures OpenAI / Groq HTTP calls on a first recording run (`--record-mode=once`
with real keys), then replays from committed YAML on every subsequent run.
Integration tests default-skip when cassettes or sample MP4 are missing —
no silent false green. Workflow doc in `tests/README.md`. `pytest.ini`
default now excludes `-m integration`.

**Step 11 — Citation grounding**
`ChatResponse` gained a `grounded: bool = False` field. `rag_generator`
parses `[M:SS]` / `(M:SS)` references out of the answer and filters
`retrieved_results` to citations within
`CITATION_TIMESTAMP_TOLERANCE_SEC` (=3 s) of a cited timestamp. Three
cases: no timestamps → ungrounded with full retrieval; timestamps with
matches → grounded with filtered list; hallucinated timestamps (no
matches) → ungrounded with full retrieval (honest surface rather than
empty citations with `grounded=True`). Prompt updated to instruct the
model to cite in `[M:SS]` format. 23 new unit tests.

**Step 12 — Frontend polish**
Added `should_poll_status` debounce keyed on `video_id` — pre-0.2 every
Streamlit rerun fired one GET per video in the sidebar, now at worst one
per 2 s per video. New "Clear uploaded videos" sidebar button scopes the
reset to client state only (backend untouched). Stored `grounded` in
chat history and renders an "Ungrounded response" caption under
assistant messages when the backend flag is False. Moved three hardcoded
timeouts into `QUADRAG_CHAT_TIMEOUT_SEC` / `QUADRAG_UPLOAD_TIMEOUT_SEC` /
`QUADRAG_STATUS_POLL_TIMEOUT_SEC` env vars.

**Step 13 — Docs + deploy smoke test**
Rewrote `CLAUDE.md` to reflect the post-refactor state. This changelog.
README spot-fixes (audio index labeled correctly as Whisper, not Gemini).

### Metrics

* **Unit tests**: 0 → 116 (adds ~2 s to the test cycle).
* **Code removed**: ~400 lines of dead / ported code.
* **Architectural indexes**: 4 advertised → 4 advertised (but now actually
  semantic, not 2 semantic + 1 keyword + 1 difflib).
* **Concurrency primitive swap**: custom sync UDFs → `pxt_openai.vision`
  (Pixeltable-native async + adaptive rate-limit throttling).
* **State safety**: 4 unlocked module dicts → 1 lock-protected store with
  atomic disk snapshots + restart replay.

---

## [0.1.0] — pre-2026-04-21 — initial prototype

Initial QuadRAG prototype. See `git log` before the 0.2.0 work for history.
