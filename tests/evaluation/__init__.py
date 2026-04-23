"""LensRAG evaluation harness.

This package contains the scripts that produce the numbers for the paper's
five placeholder tables and three case studies. All scripts operate against
the local backend venv and write their artifacts under ``results/``.

Run order:
    1. ``prep_videomme.py``        -> data/slice.json
    2. ``run_indexing_sweep.py``   -> results/indexing_times.csv
    3. ``run_eval.py``             -> results/eval_runs.jsonl + eval_summary.csv
    4. ``run_query_latency.py``    -> results/query_latency.csv + token_counts.csv
    5. ``run_case_studies.py``     -> results/case_studies.json
    6. ``writeback_paper.py``      -> /Users/sanju/Documents/code/intro/report v2.tex
"""
