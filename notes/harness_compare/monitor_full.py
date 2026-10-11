"""Print a compact, read-only progress snapshot for the full pass@4 run."""

import json
import sys
from collections import Counter
from pathlib import Path

from analyze_full import analyze


report = analyze(Path(sys.argv[1]))
errors = Counter()
examples = []
for task, group in report["by_task"].items():
    for attempt in group["attempts"]:
        if attempt["state"] != "infrastructure_failure":
            continue
        reason = (attempt["error"] or "unknown").split(":", 1)[0]
        errors[reason] += 1
        if len(examples) < 8:
            examples.append({"task": task, "sample": attempt["sample_index"], "error": attempt["error"]})
print(json.dumps({
    "states": report["states"],
    "complete_tasks": report["complete_tasks"],
    "pass_at_4_lower_bound": report["pass_at_4_lower_bound"],
    "pass_at_4_upper_bound": report["pass_at_4_upper_bound"],
    "budget_limited": report["budget_limited"],
    "error_types": dict(errors),
    "error_examples": examples,
}, indent=2))
