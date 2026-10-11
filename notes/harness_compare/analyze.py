"""Summarize the cross-harness pilot without treating missing runs as failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def dsh_projection(result_dir: Path) -> dict | None:
    paths = list((result_dir / "dsh-home/storages/session_projcache/sessions").glob("*.json"))
    if len(paths) != 1:
        return None
    value = json.loads(paths[0].read_text())
    rows = value["record"]["rows"]
    return {
        "usage": rows["tokenUsage"]["val"]["totals"],
        "last_step": rows["tokenUsage"]["val"]["last"]["step"],
    }


def claude_projection(result_dir: Path) -> dict | None:
    path = result_dir / "stdout.log"
    if not path.exists():
        return None
    results = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            results.append(event)
    if len(results) != 1:
        return None
    event = results[0]
    return {
        "is_error": event.get("is_error"),
        "model_steps": event.get("num_turns"),
        "usage": event.get("usage"),
        "model_usage": event.get("modelUsage"),
    }


def summarize(results: Path, pi_results: Path, claude_results: Path | None = None) -> dict:
    pilot = []
    for summary in sorted(results.glob("*/summary.json")):
        row = json.loads(summary.read_text())
        case = summary.parent
        row = {**row, "harness": "deepseek", "projection": dsh_projection(case)}
        pilot.append(row)

    pi_by_task: dict[str, list[dict]] = {}
    for path in sorted(pi_results.glob("*/summary.json")):
        for row in json.loads(path.read_text()):
            if row["arm"] == "baseline":
                pi_by_task.setdefault(row["task_name"], []).append(row)

    comparisons = []
    for row in pilot:
        pi = pi_by_task.get(row["task_name"], [])
        projection = row["projection"]
        comparisons.append({
            "task": row["task_name"],
            "deepseek": {
                "reward": row["reward"], "error": row["error"],
                "time_seconds": row["time_seconds"], "tool_calls": row["turns_used"],
                "usage": projection["usage"] if projection else None,
                "model_steps": projection["last_step"] if projection else None,
                "within_pi_40_step_cap": bool(projection and projection["last_step"] <= 40),
            },
            "pi_baseline": [{
                "reward": p["reward"], "error": p["error"],
                "time_seconds": p["time_seconds"], "tool_calls": p.get("tool_calls"),
                "input_tokens": p.get("input_tokens"), "output_tokens": p.get("output_tokens"),
            } for p in pi],
        })
    report = {
        "completed": len(pilot),
        "usable": sum(row["error"] is None for row in pilot),
        "pass": sum(row["reward"] == 1 and row["error"] is None for row in pilot),
        "comparisons": comparisons,
    }
    if claude_results is not None:
        claude = []
        for summary in sorted(claude_results.glob("*/summary.json")):
            row = json.loads(summary.read_text())
            case = summary.parent
            pi = pi_by_task.get(row["task_name"], [])
            claude.append({
                "task": row["task_name"],
                "reward": row["reward"],
                "error": row["error"],
                "time_seconds": row["time_seconds"],
                "tool_calls": row["turns_used"],
                "projection": claude_projection(case),
                "has_candidate": (case / "candidate.patch").exists(),
                "has_grading": (case / "grading.json").exists(),
                "pi_baseline_rewards": [p["reward"] for p in pi],
            })
        report["claude"] = {
            "completed": len(claude),
            "usable": sum(row["error"] is None and row["has_grading"] for row in claude),
            "pass": sum(row["reward"] == 1 and row["error"] is None for row in claude),
            "comparisons": claude,
        }
    return report


if __name__ == "__main__":
    report = summarize(
        Path(sys.argv[1]), Path(sys.argv[2]),
        Path(sys.argv[3]) if len(sys.argv) > 3 else None,
    )
    print(json.dumps(report, indent=2))
