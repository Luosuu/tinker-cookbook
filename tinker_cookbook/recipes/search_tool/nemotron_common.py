"""Shared helpers for the Nemotron web-search MCQA recipe.

Small utilities used by the training entrypoint, standalone eval, and split
builder: a minimal .env loader and provider-key validation. Centralized here so
they are defined once instead of copy-pasted across entrypoints.
"""

from __future__ import annotations

import os
from pathlib import Path

from tinker_cookbook.recipes.search_tool.search_providers import provider_env_var

# Default persisted train/validation split repo. Override with `dataset_name=`
# on the CLI to point at your own split (see make_split.py to create one).
DEFAULT_SPLIT_DATASET = "luosuu/nemotron-web-search-mcqa-split"


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


def require_provider_key(provider: str) -> None:
    """Raise SystemExit unless the selected provider's API key is in the env."""
    try:
        key_name = provider_env_var(provider)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if not os.environ.get(key_name):
        raise SystemExit(f"{key_name} not set (checked env and repo-root .env).")
