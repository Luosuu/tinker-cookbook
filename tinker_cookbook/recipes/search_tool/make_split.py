"""Create a stable train/validation split of the Nemotron web-search MCQA
dataset and push it to the Hugging Face Hub.

The upstream dataset (nvidia/Nemotron-RL-knowledge-web_search-mcqa) ships a
single `train` split. This script deterministically carves off a fixed
validation set and pushes both splits to a personal HF dataset repo, so
training and eval draw from a persisted, reproducible partition instead of
reconstructing "held-out" questions via matching RNG seeds.

Usage:
    uv run python -m tinker_cookbook.recipes.search_tool.make_split \
        repo_id=luosuu/nemotron-web-search-mcqa-split \
        n_val=200 seed=12345 private=False

Requires an HF write token (HF_TOKEN, loaded from repo-root .env if present).
All original columns are preserved; only the split membership is added.
"""

from __future__ import annotations

import os

import chz
from datasets import Dataset, DatasetDict, load_dataset

from tinker_cookbook.recipes.search_tool.nemotron_common import (
    DEFAULT_SPLIT_DATASET,
    load_dotenv,
    repo_root_dotenv,
)

_SOURCE = "nvidia/Nemotron-RL-knowledge-web_search-mcqa"


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
    seed: int = 12345
    private: bool = False


def main(cfg: Config) -> None:
    load_dotenv(repo_root_dotenv())
    token = _hf_token()

    full: Dataset = load_dataset(_SOURCE, split="train")
    n_total = len(full)
    if cfg.n_val >= n_total:
        raise SystemExit(f"n_val={cfg.n_val} must be < dataset size {n_total}")

    # Deterministic shuffle, then a fixed split. train_test_split with a seed is
    # reproducible, so anyone reloading the pushed repo gets the same partition.
    shuffled = full.shuffle(seed=cfg.seed)
    val = shuffled.select(range(cfg.n_val))
    train = shuffled.select(range(cfg.n_val, n_total))

    dd = DatasetDict({"train": train, "validation": val})
    print(
        f"Source {_SOURCE}: {n_total} rows -> "
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
