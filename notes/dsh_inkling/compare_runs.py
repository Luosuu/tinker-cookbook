"""Compare two harness configurations across repeated runs of the same tasks.

A single 100-task run has a 95% interval of roughly ±7 points here, and two
configuration-identical runs of this harness scored 53 and 45. Comparing one draw against
another therefore cannot credit or reject a harness change, which is what this module
exists to fix.

It pools several runs per configuration and compares them paired by task, so task
difficulty cancels instead of dominating the variance.

Usage::

    python -m notes.dsh_inkling.compare_runs \\
        --arm control=/tmp/dsh-contree/full02,/tmp/dsh-contree/full07 \\
        --arm treatment=/tmp/dsh-contree/full06
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _records(run_dir: Path) -> list[dict]:
    """Episode records from a DSH run (``results/*/summary.json``) or a native run
    (``<timestamp>/results.jsonl``, one record per line with the same fields)."""
    records: list[dict] = []
    for summary in sorted((run_dir / "results").glob("*/summary.json")):
        try:
            records.append(json.loads(summary.read_text()))
        except (OSError, ValueError):
            continue
    for results in sorted(run_dir.glob("*/results.jsonl")):
        for line in results.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def load_rewards(run_dir: Path) -> dict[str, list[float]]:
    """Per-task reward lists for one run (a list because runs may repeat a task)."""
    out: dict[str, list[float]] = defaultdict(list)
    for record in _records(run_dir):
        reward = record.get("reward")
        task = record.get("task_name")
        if task is not None and reward is not None:
            out[task].append(float(reward))
    return dict(out)


def pass_at_k(rewards: dict[str, list[float]], k: int) -> float | None:
    """Unbiased pass@k averaged over tasks; ``None`` if any task has fewer than k samples."""
    if not rewards or any(len(v) < k for v in rewards.values()):
        return None
    per_task = []
    for values in rewards.values():
        n = len(values)
        c = sum(1 for r in values if r == 1.0)
        per_task.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return _mean(per_task)


def load_arm(run_dirs: list[Path]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = defaultdict(list)
    for run_dir in run_dirs:
        for task, rewards in load_rewards(run_dir).items():
            merged[task].extend(rewards)
    return dict(merged)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare(control: dict[str, list[float]], treatment: dict[str, list[float]]) -> dict:
    """Paired-by-task comparison of mean reward per task."""
    tasks = sorted(set(control) & set(treatment))
    only_control = sorted(set(control) - set(treatment))
    only_treatment = sorted(set(treatment) - set(control))

    deltas = []
    for task in tasks:
        deltas.append(_mean(treatment[task]) - _mean(control[task]))

    n = len(deltas)
    mean_delta = _mean(deltas)
    # Paired sd of the per-task difference: task difficulty cancels within each pair.
    if n > 1:
        var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("inf")

    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = n - wins - losses
    discordant = wins + losses
    # Exact McNemar over discordant task pairs, two-sided.
    if discordant:
        k = min(wins, losses)
        p = 2 * sum(math.comb(discordant, i) for i in range(k + 1)) / 2**discordant
        p = min(p, 1.0)
    else:
        p = 1.0

    return {
        "tasks": n,
        "control_mean": _mean([_mean(v) for v in control.values()]),
        "treatment_mean": _mean([_mean(v) for v in treatment.values()]),
        "control_solved": sum(1 for v in control.values() for r in v if r == 1.0),
        "treatment_solved": sum(1 for v in treatment.values() for r in v if r == 1.0),
        "control_episodes": sum(len(v) for v in control.values()),
        "treatment_episodes": sum(len(v) for v in treatment.values()),
        "mean_delta": mean_delta,
        "se": se,
        "ci95": (mean_delta - 1.96 * se, mean_delta + 1.96 * se) if n > 1 else None,
        "tasks_only_in_control": only_control,
        "tasks_only_in_treatment": only_treatment,
        "task_wins": wins,
        "task_losses": losses,
        "task_ties": ties,
        "mcnemar_p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RUN[,RUN...]",
        help="Named configuration with one or more run directories.",
    )
    args = parser.parse_args()

    arms: dict[str, dict[str, list[float]]] = {}
    for spec in args.arm:
        name, _, dirs = spec.partition("=")
        paths = [Path(d) for d in dirs.split(",") if d]
        arms[name] = load_arm(paths)
        episodes = sum(len(v) for v in arms[name].values())
        print(f"{name}: {len(arms[name])} tasks, {episodes} episodes across {len(paths)} run(s)")
        min_samples = min((len(v) for v in arms[name].values()), default=0)
        passes = [
            f"pass@{k} {score:.3f}"
            for k in (1, 2, 4)
            if k <= min_samples and (score := pass_at_k(arms[name], k)) is not None
        ]
        if passes:
            print(f"  {', '.join(passes)}")

    names = list(arms)
    if len(names) == 2:
        control, treatment = names
        result = compare(arms[control], arms[treatment])
        print(f"\n== {control} vs {treatment} ==")
        print(f"  paired tasks: {result['tasks']}")
        print(
            f"  {control}: {result['control_solved']}/{result['control_episodes']} episodes "
            f"({result['control_mean']:.3f} mean task reward)"
        )
        print(
            f"  {treatment}: {result['treatment_solved']}/{result['treatment_episodes']} episodes "
            f"({result['treatment_mean']:.3f} mean task reward)"
        )
        print(f"  mean paired delta: {result['mean_delta']:+.3f} (sd {result['se']:.3f})")
        if result["ci95"]:
            lo, hi = result["ci95"]
            print(f"  95% CI: {lo:+.3f} to {hi:+.3f}")
            if lo <= 0 <= hi:
                print("  -> interval crosses zero: no demonstrated effect")
            else:
                print("  -> interval excludes zero")
        print(
            f"  task wins/losses/ties: {result['task_wins']}/{result['task_losses']}/"
            f"{result['task_ties']}   exact McNemar p = {result['mcnemar_p']:.4f}"
        )
        if result["tasks_only_in_control"] or result["tasks_only_in_treatment"]:
            print(
                f"  WARNING unpaired tasks: only-{control}={len(result['tasks_only_in_control'])} "
                f"only-{treatment}={len(result['tasks_only_in_treatment'])}"
            )


if __name__ == "__main__":
    main()
