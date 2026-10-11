"""Analyze a 100-task, four-sample run without hiding missing or invalid trials."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def analyze(root: Path) -> dict:
    design = json.loads((root / "design.json").read_text())
    if len(design["tasks"]) != 100 or design["samples"] != 4:
        raise ValueError("Expected the frozen 100-task, four-sample design")
    split = {
        task: group
        for group, tasks in design["splits"].items()
        for task in tasks
    }
    if set(split) != set(design["tasks"]):
        raise ValueError("Task split does not cover the design exactly")
    by_task = {}
    all_attempts = []
    for task in design["tasks"]:
        attempts = []
        for sample in range(4):
            folder = root / "results" / task / f"sample-{sample}"
            summary_path = folder / "summary.json"
            row = {
                "sample_index": sample,
                "state": "missing",
                "reward": None,
                "error": None,
                "time_seconds": None,
                "tool_calls": None,
                "model_turns": None,
                "invocation_count": None,
                "continuation_stop_reason": None,
                "budget_limited": False,
                "input_tokens_reported": None,
                "output_tokens_reported": None,
            }
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["task_name"] != task or summary["sample_index"] != sample:
                    row.update(state="infrastructure_failure", error="Summary identity mismatch")
                else:
                    status_path = folder / "status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    row.update(
                        reward=summary["reward"],
                        error=summary["error"],
                        time_seconds=summary["time_seconds"],
                        tool_calls=summary["turns_used"],
                        model_turns=status.get("model_turns_observed"),
                        invocation_count=status.get("invocation_count"),
                        continuation_stop_reason=status.get("continuation_stop_reason"),
                        budget_limited=bool(status.get("budget_limited")),
                    )
                    required = ("candidate.patch", "grading.json", "preflight.json", "status.json")
                    missing = [filename for filename in required if not (folder / filename).exists()]
                    if summary["error"] is not None or missing or status.get("bridge_errors"):
                        row["state"] = "infrastructure_failure"
                        if missing:
                            row["error"] = "Missing artifacts: " + ", ".join(missing)
                    else:
                        row["state"] = "pass" if summary["reward"] > 0 else "model_failure"
                    usage_path = folder / "claude_result.json"
                    if usage_path.exists():
                        usage = json.loads(usage_path.read_text()).get("usage") or {}
                        row["input_tokens_reported"] = sum(
                            usage.get(key, 0) or 0
                            for key in (
                                "input_tokens", "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                            )
                        )
                        row["output_tokens_reported"] = usage.get("output_tokens")
            attempts.append(row)
            all_attempts.append(row)
        valid = all(row["state"] in {"pass", "model_failure"} for row in attempts)
        successes = sum(row["state"] == "pass" for row in attempts)
        by_task[task] = {
            "split": split[task],
            "gpu": design["image"][task]["gpu"],
            "valid_samples": sum(row["state"] in {"pass", "model_failure"} for row in attempts),
            "successes": successes,
            "pass_at_4": int(successes > 0) if valid else None,
            "attempts": attempts,
        }

    complete = [row for row in by_task.values() if row["pass_at_4"] is not None]
    all_complete = len(complete) == 100
    valid_attempts = [row for row in all_attempts if row["state"] in {"pass", "model_failure"}]
    durations = [row["time_seconds"] for row in valid_attempts if row["time_seconds"] is not None]
    inputs = [row["input_tokens_reported"] for row in valid_attempts if row["input_tokens_reported"] is not None]
    outputs = [row["output_tokens_reported"] for row in valid_attempts if row["output_tokens_reported"] is not None]
    by_split = {}
    for group in ("train", "heldout"):
        tasks = [row for row in by_task.values() if row["split"] == group]
        completed = [row for row in tasks if row["pass_at_4"] is not None]
        by_split[group] = {
            "tasks": len(tasks),
            "complete_tasks": len(completed),
            "pass_at_4": (
                sum(row["pass_at_4"] for row in completed) / len(tasks)
                if len(completed) == len(tasks) else None
            ),
            "passes": sum(row["successes"] > 0 for row in tasks),
        }
    return {
        "tasks": 100,
        "expected_attempts": 400,
        "states": {state: sum(row["state"] == state for row in all_attempts) for state in (
            "pass", "model_failure", "infrastructure_failure", "missing"
        )},
        "complete_tasks": len(complete),
        "pass_at_4": sum(row["pass_at_4"] for row in complete) / 100 if all_complete else None,
        "pass_at_4_lower_bound": sum(row["successes"] > 0 for row in by_task.values()) / 100,
        "pass_at_4_upper_bound": sum(
            row["successes"] > 0 or row["pass_at_4"] is None for row in by_task.values()
        ) / 100,
        "pass_at_1": (
            sum(row["state"] == "pass" for row in all_attempts) / 400
            if len(valid_attempts) == 400 else None
        ),
        "median_time_seconds": statistics.median(durations) if durations else None,
        "median_input_tokens_reported": statistics.median(inputs) if inputs else None,
        "median_output_tokens_reported": statistics.median(outputs) if outputs else None,
        "usage_present": len(inputs),
        "budget_limited": sum(row["budget_limited"] for row in all_attempts),
        "by_split": by_split,
        "by_task": by_task,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))
