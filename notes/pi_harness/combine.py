"""Combine same-period Rivanna blocks into one treatment-vs-control test.

Primary test, paired by task: within each block, a task's difference is the treatment arm's
mean reward on it minus the control arm's; blocks are averaged per task, and the 100 task
differences get a paired t-test and a task bootstrap interval. Arms are only ever compared
within the block they ran in, so period drift cancels.

Secondary: each block's draw-level Welch delta, combined by inverse variance, with Student-t
p-values per block (block_report's normal approximation overstates significance at df~5).

Usage::

    python -m notes.pi_harness.combine --treatment pi_check --control native BLOCK [BLOCK ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

SYSTEMIC_TIMEOUT = "kokkos__kokkos-8182"


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    if math.isinf(t):
        return 0.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _episodes(run_dir: Path) -> list[dict]:
    out = []
    for path in sorted((run_dir / "results").glob("*/summary.json")):
        record = json.loads(path.read_text())
        if record.get("error") and record.get("task_name") != SYSTEMIC_TIMEOUT:
            continue
        out.append(record)
    return out


def arm_tasks(block: Path, arm: str) -> tuple[dict[str, list[float]], list[float]]:
    per_task: dict[str, list[float]] = defaultdict(list)
    draws = []
    for run_dir in sorted(block.glob(f"runs/{arm}-d*")):
        records = _episodes(run_dir)
        if records:
            draws.append(100 * sum(r.get("reward") == 1.0 for r in records) / len(records))
        for record in records:
            per_task[record["task_name"]].append(float(record.get("reward") or 0.0))
    return per_task, draws


def combine(blocks: list[Path], treatment: str, control: str, *, seed: int = 0) -> dict:
    task_deltas: dict[str, list[float]] = defaultdict(list)
    per_block = []
    for block in blocks:
        treat_tasks, treat_draws = arm_tasks(block, treatment)
        ctrl_tasks, ctrl_draws = arm_tasks(block, control)
        for task in set(treat_tasks) & set(ctrl_tasks):
            task_deltas[task].append(st.mean(treat_tasks[task]) - st.mean(ctrl_tasks[task]))
        va = st.variance(ctrl_draws) / len(ctrl_draws)
        vb = st.variance(treat_draws) / len(treat_draws)
        delta = st.mean(treat_draws) - st.mean(ctrl_draws)
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (len(ctrl_draws) - 1) + vb**2 / (len(treat_draws) - 1))
        per_block.append(
            {
                "block": block.name,
                "treatment_draws": [round(x, 1) for x in treat_draws],
                "control_draws": [round(x, 1) for x in ctrl_draws],
                "delta": round(delta, 2),
                "se": round(se, 2),
                "welch_t_p": round(t_two_sided_p(delta / se, df), 3) if se else None,
            }
        )
    diffs = [100 * st.mean(v) for v in task_deltas.values()]
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    t = mean / (sd / math.sqrt(n)) if sd else math.inf
    rng = random.Random(seed)
    boot = sorted(st.mean(rng.choices(diffs, k=n)) for _ in range(10000))
    weights = [1 / b["se"] ** 2 for b in per_block if b["se"]]
    pooled = sum(w * b["delta"] for w, b in zip(weights, per_block, strict=False)) / sum(weights)
    pooled_se = math.sqrt(1 / sum(weights))
    return {
        "treatment": treatment,
        "control": control,
        "blocks": per_block,
        "task_paired": {
            "tasks": n,
            "delta_points": round(mean, 2),
            "t": round(t, 3),
            "df": n - 1,
            "p": round(t_two_sided_p(t, n - 1), 4),
            "bootstrap_95": [round(boot[250], 2), round(boot[9749], 2)],
        },
        "draw_level_inverse_variance": {
            "delta_points": round(pooled, 2),
            "se": round(pooled_se, 2),
            "z_p": round(math.erfc(abs(pooled / pooled_se) / math.sqrt(2)), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blocks", nargs="+", type=Path)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args()
    print(json.dumps(combine(args.blocks, args.treatment, args.control), indent=2))


if __name__ == "__main__":
    main()
