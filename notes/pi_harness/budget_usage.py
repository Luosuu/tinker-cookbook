"""Measured resource use per arm, for calibrating a budget's secondary limits.

A budget's request cap should be the binding limit; every other limit should sit above
what episodes actually use. This reports, per arm, the distribution of requests, output and
input tokens, episode wall time and harness build checks, and how often each limit stopped
an episode.

Usage::

    python -m notes.pi_harness.budget_usage BLOCK
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

    return {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": ordered[-1]}


def _episode(directory: Path) -> dict | None:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("error"):
        return None
    record = {
        "turns": summary.get("turns_used") or 0,
        "stop": summary.get("stop_reason") or summary.get("harness_status"),
        "output": summary.get("output_tokens"),
        "input": summary.get("input_tokens"),
        "seconds": summary.get("time_seconds"),
        "checks": None,
        "check_seconds": None,
    }
    result = directory / "pi-result.json"
    if result.is_file():
        stats = json.loads(result.read_text()).get("build_check_stats")
        if stats:
            record["checks"], record["check_seconds"] = stats["runs"], stats["seconds"]
    return record


def usage(block: Path) -> dict:
    arms: dict[str, list[dict]] = defaultdict(list)
    for directory in sorted(block.glob("runs/*/results/*")):
        record = _episode(directory)
        if record is not None:
            arms[directory.parents[1].name.rsplit("-d", 1)[0]].append(record)
    out = {}
    for arm, records in sorted(arms.items()):
        out[arm] = {
            "episodes": len(records),
            "stop_reasons": dict(Counter(r["stop"] for r in records).most_common()),
            **{
                key: _quantiles([r[key] for r in records if r[key] is not None])
                for key in ("turns", "output", "input", "seconds", "checks", "check_seconds")
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", type=Path)
    print(json.dumps(usage(parser.parse_args().block), indent=2))


if __name__ == "__main__":
    main()
