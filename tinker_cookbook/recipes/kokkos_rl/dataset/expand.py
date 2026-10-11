"""Unified entry point: ``expand mine ...`` or ``expand run ...``.

Mine public candidates locally; run uses local Chat Completions inference and
credential-free Rivanna Slurm/Apptainer workers. See dataset/README.md for flags.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"mine", "run"}:
        raise SystemExit(
            "Usage: python -m tinker_cookbook.recipes.kokkos_rl.dataset.expand {mine|run} [options]"
        )
    action = sys.argv.pop(1)
    if action == "mine":
        from .expansion import main as run
    else:
        from .expansion_batch import main as run
    run()


if __name__ == "__main__":
    main()
