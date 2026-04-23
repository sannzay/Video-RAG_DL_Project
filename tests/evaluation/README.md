# LensRAG evaluation harness

Fills the five placeholder tables and three case-study blocks in
`/Users/sanju/Documents/code/intro/report v1.tex` with measured numbers
drawn from a stratified Video-MME subset.

## Prerequisites

1. The backend venv and `.env`:
   ```bash
   cd backend && source .venv/bin/activate && cd ..
   ```
   `backend/.env` must carry `OPENAI_API_KEY` (Whisper + embeddings) and
   `OPENROUTER_API_KEY` (Llama 3.3 70B + Gemini 2.0 Flash).

2. The Video-MME subset staged at
   `/Users/sanju/Documents/code/intro/Video_RAG/`:
   * `videomme_subset_qa.jsonl`  — 270 QA items.
   * `videomme_subset_videos.csv` — 89 video rows.
   * `videorag/*.mp4`            — 88 MP4s.
   * `videorag/*.webm` (optional) — transcoded to MP4 on demand.

## Run order

Strict order — each phase reads the previous phase's artifacts.

```
# 1. Build the 15-video stratified slice (5 short + 5 medium + 5 long)
python tests/evaluation/prep_videomme.py

# 2. Index every slice video; time each of the four indexes
python tests/evaluation/run_indexing_sweep.py

# 3. 5-configuration ablation: audio-only, image-only, audio+image,
#    +description, full four-index. Same slice across all five.
python tests/evaluation/run_eval.py

# 4. Query-latency + token-cost sweep on one dialogue-heavy and one
#    visual-heavy archetype video (each N=100 queries).
python tests/evaluation/run_query_latency.py

# 5. Pick three qualitative case studies from the ablation results.
python tests/evaluation/run_case_studies.py

# 6. Replace `---` placeholder cells in report v1.tex and write v2.
python tests/evaluation/writeback_paper.py
```

All scripts are idempotent. Re-running a phase picks up where it left off
(Phase 2 skips videos already in the registry; Phase 3 skips
`(config, question_id)` pairs already recorded). Pass `--force` to
regenerate from scratch.

## Artifacts

```
tests/evaluation/
├── data/
│   └── slice.json                   # Stratified 15-video subset + 45 QA items
├── results/
│   ├── indexing_times.csv           # Per-video per-stage wall-clock
│   ├── indexing_provenance.json     # Per-video lens + status
│   ├── indexing_sweep.log           # Raw log of the sweep
│   ├── eval_runs.jsonl              # One line per (config, QA) result
│   ├── eval_summary.csv             # Per-config Top-1, grounded rate, CI
│   ├── query_latency.csv            # Per-query stage timings
│   ├── query_latency_summary.csv    # Per-stage mean/median/p95
│   ├── token_counts.csv             # Per-query prompt + completion tokens
│   └── case_studies.json            # 3 archetype picks with full context
└── (report v2.tex lives at /Users/sanju/Documents/code/intro/report v2.tex)
```

## Smoke tests & safety checks

* `run_eval.py --configs full --limit 1 --skip-missing` — one QA through
  the full config; confirms search + fusion + generation + letter
  extraction work end-to-end before a multi-hour run.
* `run_eval.py` asserts `ResultFusion()` observes the patched weight
  vector at the top of every config block; writes a `config_start`
  provenance line into the JSONL.
* `writeback_paper.py` warns (and names the label) if any `---` remains
  inside a `\begin{tabular}` block after the substitution pass.

## Knobs that matter

* `SEED=42` in `prep_videomme.py` — change to resample the slice.
* `PER_BUCKET=5` in `prep_videomme.py` — 5 per duration bucket.
* `ABLATION_CONFIGS` in `_common.py` — weight vectors for the 5 configs.
* `CHAT_MODEL_{INPUT,OUTPUT}_PER_M` in both `run_query_latency.py` and
  `writeback_paper.py` — update when OpenRouter pricing changes.
* `NO_LENS_WEIGHTS` in `run_case_studies.py` — the synthesized weights
  used to rerun an archetype-3 candidate without the lens leg.

## Known constraints

* The image index was built with `image_embed` only (no `string_embed`).
  `run_eval.py` and `run_case_studies.py` therefore route text queries
  through CLIP's text tower manually via `search_image_index_by_text`;
  frame embeddings are cached per-video after the first call.
* Some videos in the slice may produce zero audio hits (music-only or
  heavy-accent clips where Whisper yields empty transcripts). Those
  cases are recorded in `eval_runs.jsonl` with `retrieved_counts.audio=0`
  and the harness continues.
* The grounded rate for MC questions is typically 0 because the prompt
  asks for a single letter, not `[M:SS]` timestamps. That's expected and
  fine for the MC accuracy metric.
