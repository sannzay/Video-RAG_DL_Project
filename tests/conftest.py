"""Shared pytest configuration for the QuadRAG test suite.

Puts ``backend/src`` on ``sys.path`` so ``import quadrag`` works without needing
the backend package to be pip-installed. Also short-circuits Settings by
feeding empty-string API keys so the Pydantic Settings instance doesn't
complain at import time when ``.env`` is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = REPO_ROOT / "backend" / "src"

if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

# Provide harmless defaults so Settings() never refuses to load in CI.
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
