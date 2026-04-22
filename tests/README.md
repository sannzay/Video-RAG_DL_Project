# QuadRAG test suite

Two tiers, run with different commands.

## Unit tests — default

Fast (<5 s), hermetic, no network, no API keys, no sample video.

```bash
backend/.venv/bin/python -m pytest
```

This is what runs in CI and what you should run before every commit.

`pytest.ini` excludes the `integration` marker by default, so the 93 tests
under `tests/unit/` are everything that runs.

## Integration tests — opt-in

Drive the full indexing + chat pipeline against a real video. They use
[pytest-recording](https://pypi.org/project/pytest-recording/) (a VCR wrapper)
so the one-time recording uses real API keys but every subsequent run replays
the committed cassettes for free.

### Required setup

1. **A sample video**: `tests/fixtures/sample.mp4` ships with the repo
   (~3 MB Kandima Maldives clip, 20 s, H.264 Main + AAC). Override with a
   different fixture by exporting `BENCHMARK_VIDEO=/path/to/your.mp4`.

2. **Real API keys in your shell** for the recording run:
   ```bash
   export OPENAI_API_KEY=sk-...
   export GROQ_API_KEY=gsk-...
   ```
   Replay runs do not need these.

### One-time recording

```bash
cd backend && ./.venv/bin/python -m pytest ../tests/integration \
    -m integration \
    --record-mode=once
```

This runs the full pipeline against real APIs (Whisper, GPT-4o-mini vision,
OpenAI embeddings, Groq chat) and writes YAML cassettes into
`tests/integration/cassettes/`. Expect 1–3 minutes per test.

**Commit the `.yaml` files** once they look reasonable — they are the
replay fixtures that future runs (and CI) depend on. The YAML filter in
`conftest.py` strips `Authorization` / `x-api-key` headers, so nothing
secret should leak in. Eyeball a cassette before committing to confirm.

### Replay (the default for everyone else)

```bash
cd backend && ./.venv/bin/python -m pytest ../tests/integration -m integration
```

With cassettes committed, this runs in seconds and needs no API keys. If a
cassette is missing, the matching test **skips with a clear message** (it
does not silently make a real API call).

### When to re-record

Re-record cassettes whenever:
- You change a prompt sent to OpenAI or Groq
- You bump a model name in `config.py`
- You change the request shape (e.g., add tools to chat completion)
- You upgrade Pixeltable or the `openai`/`groq` SDKs to a version that
  changes request bodies

Delete the stale cassette and re-run the recording command above.

### Troubleshooting

- **`pytest.skip`: "Integration tests need a sample MP4..."** — drop a file
  at `tests/fixtures/sample.mp4`.
- **`pytest.skip`: "No cassette at ..."** — either record it (see above) or
  you're expected to skip this one for now.
- **`CassetteNotFoundError`** at replay time — the VCR matcher couldn't
  find a matching request. Usually means the test changed the request
  shape; re-record.
- **`ValueError: Domain view name ... exceeds Postgres identifier limit`**
  — you have a Pixeltable directory name that's too long. Shorten the
  video UUID prefix in the test fixture.

## Directory layout

```
tests/
├── conftest.py                  shared: puts backend/src on sys.path
├── fixtures/
│   └── sample.mp4               gitignored; required for integration
├── integration/
│   ├── cassettes/               committed VCR recordings (YAML)
│   ├── conftest.py              fixtures + VCR config
│   ├── test_chat_endpoint.py    /chat end-to-end
│   └── test_indexing_pipeline.py  all four indexes build correctly
└── unit/                        fast, hermetic, always-run
    ├── test_domain_view_registry.py
    ├── test_extract_text_from_chunk.py
    ├── test_frame_count.py
    ├── test_fusion.py
    ├── test_processing_state.py
    ├── test_registry_round_trip.py
    └── test_state_persistence.py
```
