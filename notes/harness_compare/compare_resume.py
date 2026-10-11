"""Paired comparison of the frozen one-shot and thinking-continuation runs."""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from analyze_full import analyze
from summarize_full import _is_production_source


def _bootstrap_interval(values: list[float], seed: int = 20260926) -> list[float]:
    random_state = random.Random(seed)
    samples = []
    for _ in range(10000):
        samples.append(sum(random_state.choices(values, k=len(values))) / len(values))
    samples.sort()
    return [samples[249], samples[9749]]


def _source_edited(folder: Path) -> bool:
    patch = folder / "candidate.patch"
    if not patch.exists():
        return False
    return any(_is_production_source(line) for line in patch.read_text(errors="replace").splitlines())


def compare(baseline: Path, continuation: Path) -> dict:
    old_design = json.loads((baseline / "design.json").read_text())
    new_design = json.loads((continuation / "design.json").read_text())
    for key in (
        "tasks", "splits", "samples", "model", "weights", "image", "task_digest",
        "max_model_turns", "max_tool_calls", "max_wall_seconds", "claude_package_lock_sha256",
    ):
        if old_design[key] != new_design[key]:
            raise ValueError(f"Frozen designs differ in {key}")
    if new_design.get("continuation_policy") != "thinking_only_resume":
        raise ValueError("New run is not the thinking-only continuation profile")

    old = analyze(baseline)
    new = analyze(continuation)
    result = {
        "sample_pairing_note": (
            "Task IDs are paired; sample indices match by label but stochastic "
            "model draws were not seed-coupled. Attempt transitions are descriptive."
        ),
        "baseline_states": old["states"],
        "continuation_states": new["states"],
        "baseline_complete_tasks": old["complete_tasks"],
        "continuation_complete_tasks": new["complete_tasks"],
        "baseline_pass_at_4": old["pass_at_4"],
        "continuation_pass_at_4": new["pass_at_4"],
        "baseline_pass_at_1": old["pass_at_1"],
        "continuation_pass_at_1": new["pass_at_1"],
    }
    eligible_tasks = [
        task for task in old_design["tasks"]
        if old["by_task"][task]["pass_at_4"] is not None
        and new["by_task"][task]["pass_at_4"] is not None
    ]
    excluded_tasks = [task for task in old_design["tasks"] if task not in eligible_tasks]
    result["paired_complete_tasks"] = len(eligible_tasks)
    result["paired_excluded_tasks"] = excluded_tasks
    result["continuation_pass_at_4_bounds_all_100"] = [
        new["pass_at_4_lower_bound"], new["pass_at_4_upper_bound"]
    ]
    if not eligible_tasks:
        result["paired_comparison_available"] = False
        return result

    task_differences = []
    attempt_differences_by_task = []
    task_outcomes = Counter()
    attempt_outcomes = Counter()
    edited = Counter()
    resumed = Counter()
    stop_reasons = Counter()
    duration = {"baseline": [], "continuation": []}
    inputs = {"baseline": [], "continuation": []}
    outputs = {"baseline": [], "continuation": []}
    calls = {"baseline": [], "continuation": []}
    turns = {"baseline": [], "continuation": []}

    by_split = {group: {"tasks": 0, "baseline_passes": 0, "continuation_passes": 0,
                        "baseline_attempt_passes": 0, "continuation_attempt_passes": 0}
                for group in ("train", "heldout")}
    for task in eligible_tasks:
        old_task = old["by_task"][task]
        new_task = new["by_task"][task]
        group = old_task["split"]
        by_split[group]["tasks"] += 1
        old_pass = bool(old_task["pass_at_4"])
        new_pass = bool(new_task["pass_at_4"])
        by_split[group]["baseline_passes"] += int(old_pass)
        by_split[group]["continuation_passes"] += int(new_pass)
        task_outcomes[("pass" if old_pass else "fail") + "_to_" +
                      ("pass" if new_pass else "fail")] += 1
        task_differences.append(int(new_pass) - int(old_pass))
        attempt_difference = 0
        for sample in range(4):
            old_row = old_task["attempts"][sample]
            new_row = new_task["attempts"][sample]
            old_success = old_row["state"] == "pass"
            new_success = new_row["state"] == "pass"
            by_split[group]["baseline_attempt_passes"] += int(old_success)
            by_split[group]["continuation_attempt_passes"] += int(new_success)
            attempt_difference += int(new_success) - int(old_success)
            attempt_outcomes[("pass" if old_success else "fail") + "_to_" +
                             ("pass" if new_success else "fail")] += 1
            for label, root, row in (
                ("baseline", baseline, old_row),
                ("continuation", continuation, new_row),
            ):
                folder = root / "results" / task / f"sample-{sample}"
                if _source_edited(folder):
                    edited[label] += 1
                duration[label].append(row["time_seconds"])
                calls[label].append(row["tool_calls"])
                turns[label].append(row["model_turns"])
                if row["input_tokens_reported"] is not None:
                    inputs[label].append(row["input_tokens_reported"])
                if row["output_tokens_reported"] is not None:
                    outputs[label].append(row["output_tokens_reported"])
            if (new_row["invocation_count"] or 0) > 1:
                resumed["attempts"] += 1
                resumed["additional_invocations"] += new_row["invocation_count"] - 1
                resumed["passed"] += int(new_success)
                resumed["production_edits"] += int(_source_edited(
                    continuation / "results" / task / f"sample-{sample}"
                ))
            stop_reasons[new_row["continuation_stop_reason"] or "missing"] += 1
        attempt_differences_by_task.append(attempt_difference / 4)

    result.update({
        "paired_comparison_available": True,
        "paired_by_split": by_split,
        "paired_baseline_pass_at_4": sum(v["baseline_passes"] for v in by_split.values()) / len(eligible_tasks),
        "paired_continuation_pass_at_4": sum(v["continuation_passes"] for v in by_split.values()) / len(eligible_tasks),
        "paired_baseline_pass_at_1": sum(v["baseline_attempt_passes"] for v in by_split.values()) / (4 * len(eligible_tasks)),
        "paired_continuation_pass_at_1": sum(v["continuation_attempt_passes"] for v in by_split.values()) / (4 * len(eligible_tasks)),
        "pass_at_4_delta": statistics.mean(task_differences),
        "pass_at_4_delta_task_bootstrap_95": _bootstrap_interval(task_differences),
        "pass_at_1_delta": statistics.mean(attempt_differences_by_task),
        "pass_at_1_delta_task_bootstrap_95": _bootstrap_interval(attempt_differences_by_task),
        "task_transitions": dict(task_outcomes),
        "attempt_transitions": dict(attempt_outcomes),
        "production_edit_attempts": dict(edited),
        "resumption": dict(resumed),
        "continuation_stop_reasons": dict(stop_reasons),
        "telemetry": {
            label: {
                "median_wall_seconds": statistics.median(duration[label]),
                "median_tool_calls": statistics.median(calls[label]),
                "median_model_turns": statistics.median(turns[label]),
                "sum_input_tokens_reported": sum(inputs[label]),
                "sum_output_tokens_reported": sum(outputs[label]),
                "usage_present": len(inputs[label]),
            }
            for label in ("baseline", "continuation")
        },
    })
    return result


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
