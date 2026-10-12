"""Pin the Kokkos dependency of historical kokkos-kernels candidates.

The pin is the last kokkos/kokkos ``develop`` commit at or before the kernels base commit,
i.e. the dependency kernels CI built against at the time. Preflight and construction still
establish compatibility by execution; an unresolvable candidate keeps no pin and stays
quarantined by the profile gate.

    python -m tinker_cookbook.recipes.kokkos_rl.dataset.pin_dependency \\
        --input kernels/candidates.jsonl --output kernels/pinned.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .expansion import rows, write_rows
from .mine import GitHubAPI


async def dependency_commit(api, base_commit: str) -> str:
    commit = await api.json(f"/repos/kokkos/kokkos-kernels/commits/{base_commit}")
    date = commit["commit"]["committer"]["date"]
    history = await api.json(
        "/repos/kokkos/kokkos/commits", {"sha": "develop", "until": date, "per_page": 1}
    )
    return history[0]["sha"]


async def pin(candidates: list[dict], api) -> list[dict]:
    pinned = []
    for row in candidates:
        metadata = row.get("metadata", {})
        if row.get("repo") == "kokkos/kokkos-kernels" and not metadata.get(
            "kokkos_dependency_commit"
        ):
            sha = await dependency_commit(api, row["base_commit"])
            row = {
                **row,
                "metadata": {
                    **metadata,
                    "kokkos_dependency_commit": sha,
                    "kokkos_dependency_rule": "kokkos develop at kernels base commit date",
                },
            }
        pinned.append(row)
    return pinned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = GitHubAPI(os.environ.get("GITHUB_TOKEN"))
    write_rows(args.output, asyncio.run(pin(rows(args.input), api)))


if __name__ == "__main__":
    main()
