"""Create a stable train/validation split of the Nemotron science dataset and
push it to the Hugging Face Hub.

The upstream dataset (nvidia/Nemotron-RL-Science-v1) ships a single `so_openq`
split of ~151k open-ended science questions. This carves off a fixed validation
set and pushes both splits to a personal HF repo for reproducible train/eval.

No content filtering is applied: the small fraction of questions needing
numerical computation are handled at train time by the in-process calculator
tool (see calculator.py), so we keep the full dataset.

Usage:
    uv run python -m tinker_cookbook.recipes.nemotron_science.make_split \
        repo_id=luosuu/nemotron-science-split n_val=200 seed=12345 private=False

Requires an HF write token (HF_TOKEN, loaded from repo-root .env if present).
"""

from __future__ import annotations

import os

import chz
from datasets import Dataset, DatasetDict, load_dataset

from tinker_cookbook.recipes.nemotron_science.common import (
    DEFAULT_SPLIT_DATASET,
    load_dotenv,
    repo_root_dotenv,
)

_SOURCE = "nvidia/Nemotron-RL-Science-v1"
_SOURCE_SPLIT = "so_openq"


def _hf_token() -> str:
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not tok:
        raise SystemExit("No HF token found (set HF_TOKEN in env or repo-root .env).")
    return tok


@chz.chz
class Config:
    repo_id: str = DEFAULT_SPLIT_DATASET
    n_val: int = 200
    n_total: int | None = None  # cap total rows taken from source (None = all)
    seed: int = 12345
    private: bool = False


def main(cfg: Config) -> None:
    load_dotenv(repo_root_dotenv())
    token = _hf_token()

    full: Dataset = load_dataset(_SOURCE, split=_SOURCE_SPLIT)
    if cfg.n_total is not None:
        full = full.select(range(min(cfg.n_total, len(full))))
    n_total = len(full)
    if cfg.n_val >= n_total:
        raise SystemExit(f"n_val={cfg.n_val} must be < dataset size {n_total}")

    shuffled = full.shuffle(seed=cfg.seed)
    val = shuffled.select(range(cfg.n_val))
    train = shuffled.select(range(cfg.n_val, n_total))

    dd = DatasetDict({"train": train, "validation": val})
    print(
        f"Source {_SOURCE}:{_SOURCE_SPLIT}: {n_total} rows -> "
        f"train={len(train)}, validation={len(val)} (seed={cfg.seed})"
    )
    print(f"Columns preserved: {full.column_names}")

    dd.push_to_hub(
        cfg.repo_id,
        private=cfg.private,
        token=token,
        commit_message=(
            f"Stable train/validation split of {_SOURCE} "
            f"(n_val={cfg.n_val}, seed={cfg.seed})"
        ),
    )
    vis = "private" if cfg.private else "public"
    print(f"\nPushed {vis} dataset to https://huggingface.co/datasets/{cfg.repo_id}")
    print("Splits: train, validation")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
