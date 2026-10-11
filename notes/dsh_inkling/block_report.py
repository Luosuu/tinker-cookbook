"""Same-period block report: pass@k, per-draw solves, and a draw-level Welch test.

Scores on this benchmark drift between sessions by several points, so arms are only
comparable when every draw of every arm runs in the same period. This reads several run
directories per arm (DSH ``results/*/summary.json`` or native ``*/results.jsonl``) and reports,
for each arm, pass@1/2/4 and the per-draw solve counts, then for each arm pair the paired-by-task
delta (reusing :mod:`compare_runs`) and a draw-level Welch t-test that treats each 100-task draw
as one observation.

Usage::

    python -m notes.dsh_inkling.block_report \
        --arm native=/tmp/dsh-contree/native16,/tmp/dsh-contree/native17 \
        --arm state=/tmp/dsh-contree/full64,/tmp/dsh-contree/full65 \
        --arm state_eb=/tmp/dsh-contree/full60,/tmp/dsh-contree/full61

Draw-level p-values use a normal approximation and should be pooled across blocks, never across
periods silently; see RESULTS.md.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

from notes.dsh_inkling.compare_runs import _records, compare, load_arm, pass_at_k

SYSTEMIC_TIMEOUT = "kokkos__kokkos-8182"


def draw_solve_rates(run_dir: Path, *, drop_infra: bool) -> float:
    """Percent solved in one run directory, optionally dropping infra-error episodes."""
    records = _records(run_dir)
    kept = [
        r
        for r in records
        if not (drop_infra and r.get("error") and r.get("task_name") != SYSTEMIC_TIMEOUT)
    ]
    if not kept:
        return 0.0
    return 100 * sum(1 for r in kept if r.get("reward") == 1.0) / len(kept)


def welch(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """Return (mean_b - mean_a, t, df, two-sided p) by Welch's approximation."""
    if len(a) < 2 or len(b) < 2:
        return st.mean(b) - st.mean(a), float("nan"), float("nan"), float("nan")
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    delta = st.mean(b) - st.mean(a)
    if se == 0:
        return delta, float("inf"), float("nan"), 0.0
    t = delta / se
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    # Normal approximation to the two-sided p-value.
    p = math.erfc(abs(t) / math.sqrt(2))
    return delta, t, df, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=RUN[,RUN...]")
    parser.add_argument(
        "--drop-infra",
        action="store_true",
        help="Exclude episodes whose summary records an infrastructure error "
        f"(except the known systemic {SYSTEMIC_TIMEOUT} verifier timeout).",
    )
    args = parser.parse_args()

    arms: dict[str, list[Path]] = {}
    for spec in args.arm:
        name, _, dirs = spec.partition("=")
        arms[name] = [Path(d) for d in dirs.split(",") if d]

    rewards = {name: load_arm(dirs) for name, dirs in arms.items()}
    draws = {
        name: [draw_solve_rates(d, drop_infra=args.drop_infra) for d in dirs]
        for name, dirs in arms.items()
    }

    for name in arms:
        rw = rewards[name]
        passes = " ".join(
            f"pass@{k}={pass_at_k({t: v for t, v in rw.items() if len(v) >= k}, k):.3f}"
            for k in (1, 2, 4)
            if min((len(v) for v in rw.values()), default=0) >= k
        )
        rounded = [round(x, 1) for x in draws[name]]
        print(f"{name:12s} draws={rounded} mean={st.mean(draws[name]):.1f} {passes}")

    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            r = compare(rewards[a], rewards[b])
            lo, hi = r["ci95"] if r["ci95"] else (float("nan"), float("nan"))
            delta, t, df, p = welch(draws[a], draws[b])
            print(
                f"\n{b} vs {a}:"
                f"\n  paired pass@1 {100 * r['mean_delta']:+.1f} "
                f"(CI {100 * lo:+.1f}..{100 * hi:+.1f}), "
                f"W/L {r['task_wins']}/{r['task_losses']}, McNemar p={r['mcnemar_p']:.3f}"
                f"\n  draw-level {delta:+.1f} t={t:.2f} df={df:.1f} p={p:.3f}"
            )


if __name__ == "__main__":
    main()
