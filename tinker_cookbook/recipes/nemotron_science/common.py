"""Shared helpers for the Nemotron science RL recipe."""

from __future__ import annotations

import os
from pathlib import Path

# Default persisted train/validation split repo (pushed by make_split.py).
# Override with `dataset_name=` on the CLI to use your own split.
DEFAULT_SPLIT_DATASET = "luosuu/nemotron-science-split"


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file (no dependency).

    Existing environment variables are not overwritten.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def repo_root_dotenv() -> Path:
    """Path to the repo-root .env (…/tinker-cookbook/.env)."""
    return Path(__file__).resolve().parents[3] / ".env"
