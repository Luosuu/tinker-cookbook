"""Fail a Slurm smoke job unless the full Claude/sandbox/grader path ran."""

import json
import sys
from pathlib import Path


folder = Path(sys.argv[1])
required = (
    "summary.json", "status.json", "preflight.json", "events.jsonl",
    "stdout.log", "stderr.log", "claude_result.json", "candidate.patch", "grading.json",
)
missing = [name for name in required if not (folder / name).exists()]
if missing:
    raise SystemExit("Smoke missing artifacts: " + ", ".join(missing))
summary = json.loads((folder / "summary.json").read_text())
status = json.loads((folder / "status.json").read_text())
if summary["error"] or status["bridge_errors"] or status["model_turns_observed"] < 1:
    raise SystemExit("Smoke had a model, bridge, or grader error")
if status["tool_calls"] < 1 or summary["turns_used"] != status["tool_calls"]:
    raise SystemExit("Smoke did not prove MCP sandbox tool routing")
if json.loads((folder / "claude_result.json").read_text())["is_error"]:
    raise SystemExit("Claude Code returned an error")
print(json.dumps({
    "task": summary["task_name"], "sample": summary["sample_index"],
    "reward": summary["reward"], "tools": status["tool_calls"],
    "model_turns": status["model_turns_observed"],
}))
