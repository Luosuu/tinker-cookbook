"""Produce compact, reproducible metrics for the frozen Claude Code pass@4 run."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from analyze_full import analyze


PATCH_PATH = re.compile(r"^diff --git a/(\S+) b/")
SOURCE_SUFFIXES = (".h", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".inl")


def _is_production_source(line: str) -> bool:
    match = PATCH_PATH.match(line)
    if not match:
        return False
    path = match.group(1)
    parts = path.split("/")
    return (
        path.endswith(SOURCE_SUFFIXES)
        and parts[0] not in {"workspace", "build", "tests", "doc", "example", "benchmarks"}
        and not any(part in {"unit_test", "test", "tests"} for part in parts)
    )


def _median(values: list[float | int]) -> float | None:
    return statistics.median(values) if values else None


def _summarize_group(tasks: list[dict]) -> dict:
    attempts = [attempt for task in tasks for attempt in task["attempts"]]
    valid = [attempt for attempt in attempts if attempt["state"] in {"pass", "model_failure"}]
    passed = sum(attempt["state"] == "pass" for attempt in valid)
    complete = [task for task in tasks if task["pass_at_4"] is not None]
    return {
        "tasks": len(tasks),
        "complete_tasks": len(complete),
        "attempts": len(attempts),
        "valid_attempts": len(valid),
        "passed_attempts": passed,
        "pass_at_1": passed / len(attempts) if len(valid) == len(attempts) else None,
        "passed_tasks": sum(task["pass_at_4"] for task in complete),
        "pass_at_4": (
            sum(task["pass_at_4"] for task in complete) / len(tasks)
            if len(complete) == len(tasks) else None
        ),
    }


def summarize(root: Path) -> dict:
    analysis = analyze(root)
    design = json.loads((root / "design.json").read_text())
    tasks = analysis["by_task"]
    attempts = [attempt for task in tasks.values() for attempt in task["attempts"]]
    valid = [attempt for attempt in attempts if attempt["state"] in {"pass", "model_failure"}]
    durations = [attempt["time_seconds"] for attempt in valid]
    calls = [attempt["tool_calls"] for attempt in valid]
    turns = [attempt["model_turns"] for attempt in valid if attempt["model_turns"] is not None]
    inputs = [attempt["input_tokens_reported"] for attempt in valid if attempt["input_tokens_reported"] is not None]
    outputs = [attempt["output_tokens_reported"] for attempt in valid if attempt["output_tokens_reported"] is not None]
    edit_outcomes: Counter[str] = Counter()
    final_content: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    malformed_tool_text = 0
    for task_name in design["tasks"]:
        for sample in range(4):
            folder = root / "results" / task_name / f"sample-{sample}"
            patch = (folder / "candidate.patch").read_text(errors="replace")
            edited = any(_is_production_source(line) for line in patch.splitlines())
            state = tasks[task_name]["attempts"][sample]["state"]
            edit_outcomes[("production_edit" if edited else "no_production_edit") + ":" + state] += 1
            assistant = []
            seen_tool_ids: set[str] = set()
            for line in (folder / "stdout.log").read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", [])
                    assistant.append(content)
                    for block in content:
                        if block.get("type") != "tool_use":
                            continue
                        tool_id = block.get("id") or json.dumps(block, sort_keys=True)
                        if tool_id not in seen_tool_ids:
                            seen_tool_ids.add(tool_id)
                            tool_names[block.get("name", "")] += 1
            final = assistant[-1] if assistant else []
            final_content["+".join(block.get("type", "unknown") for block in final) or "empty"] += 1
            malformed_tool_text += int(any(
                block.get("type") == "text" and "mcp__hpc__bash" in block.get("text", "")
                for block in final
            ))

    grouped = {
        "overall": _summarize_group(list(tasks.values())),
        "train": _summarize_group([task for task in tasks.values() if task["split"] == "train"]),
        "heldout": _summarize_group([task for task in tasks.values() if task["split"] == "heldout"]),
        "cpu": _summarize_group([task for task in tasks.values() if not task["gpu"]]),
        "gpu": _summarize_group([task for task in tasks.values() if task["gpu"]]),
    }
    return {
        "states": analysis["states"],
        "groups": grouped,
        "budget_limited": analysis["budget_limited"],
        "telemetry": {
            "median_wall_seconds": _median(durations),
            "total_wall_hours": sum(durations) / 3600,
            "median_tool_calls": _median(calls),
            "max_tool_calls": max(calls) if calls else None,
            "total_tool_calls": sum(calls),
            "median_model_turns": _median(turns),
            "max_model_turns": max(turns) if turns else None,
            "reported_token_usage_present": len(inputs),
            "median_input_tokens_reported": _median(inputs),
            "median_output_tokens_reported": _median(outputs),
            "sum_input_tokens_reported": sum(inputs),
            "sum_output_tokens_reported": sum(outputs),
        },
        "candidate_edit_outcomes": dict(sorted(edit_outcomes.items())),
        "final_assistant_content": dict(sorted(final_content.items())),
        "malformed_tool_text": malformed_tool_text,
        "model_tool_names": dict(sorted(tool_names.items())),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
