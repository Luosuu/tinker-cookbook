"""Per-arm tool-use mechanics for Pi episodes in a Rivanna block.

For each Pi arm: tool mix, characters returned per read, edit success, how many requests
the episode used, and when it first changed the repository. These are the mechanism
metrics for the file-tools lever; the score is read separately with ``block_report``.

Usage::

    python -m notes.pi_harness.tool_use BLOCK
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

READ_COMMAND = re.compile(r"^\s*(cat|sed -n|head|tail|nl|awk)\b")
EDIT_COMMAND = re.compile(r"sed\s+-i|git\s+apply|patch\s+-|cat\s*>|tee\s|python3?\s+-")


def _text(result: dict) -> str:
    return "".join(c.get("text", "") for c in (result or {}).get("content", []))


def episode_tool_use(events_path: Path) -> dict:
    starts: dict[str, dict] = {}
    calls = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line).get("event") or {}
        if event.get("type") == "tool_execution_start":
            starts[event["toolCallId"]] = event
        elif event.get("type") == "tool_execution_end":
            start = starts.get(event["toolCallId"], {})
            calls.append(
                {
                    "tool": event["toolName"],
                    "args": start.get("args") or {},
                    "error": bool(event.get("isError")),
                    "chars": len(_text(event.get("result"))),
                }
            )

    def is_read(call: dict) -> bool:
        return call["tool"] == "read" or (
            call["tool"] == "bash" and bool(READ_COMMAND.search(call["args"].get("command", "")))
        )

    def is_edit(call: dict) -> bool:
        return call["tool"] == "edit" or (
            call["tool"] == "bash" and bool(EDIT_COMMAND.search(call["args"].get("command", "")))
        )

    first_edit = next((i for i, c in enumerate(calls) if is_edit(c)), None)
    return {
        "calls": len(calls),
        "tools": Counter(c["tool"] for c in calls),
        "read_chars": [c["chars"] for c in calls if is_read(c)],
        "edits": sum(c["tool"] == "edit" for c in calls),
        "edit_errors": sum(c["tool"] == "edit" and c["error"] for c in calls),
        "first_edit": first_edit,
    }


def summarize(block: Path) -> dict:
    arms: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for summary_path in sorted(block.glob("runs/*/results/*/summary.json")):
        directory = summary_path.parent
        events = directory / "events.jsonl"
        if not events.is_file():
            continue
        arm = directory.parents[1].name.rsplit("-d", 1)[0]
        arms[arm].append((json.loads(summary_path.read_text()), episode_tool_use(events)))
    out = {}
    for arm, items in sorted(arms.items()):
        valid = [(s, u) for s, u in items if not s.get("error")]
        reads = [c for _, u in valid for c in u["read_chars"]]
        tools: Counter = Counter()
        for _, u in valid:
            tools.update(u["tools"])
        first = [u["first_edit"] for _, u in valid if u["first_edit"] is not None]
        edits = sum(u["edits"] for _, u in valid)
        out[arm] = {
            "episodes": len(valid),
            "solved": sum(s.get("reward") == 1.0 for s, _ in valid),
            "tool_mix": dict(tools.most_common()),
            "reads": len(reads),
            "median_read_chars": statistics.median(reads) if reads else None,
            "edit_tool_calls": edits,
            "edit_tool_error_rate": round(sum(u["edit_errors"] for _, u in valid) / edits, 2)
            if edits
            else None,
            "median_first_edit_call": statistics.median(first) if first else None,
            "never_edited": sum(u["first_edit"] is None for _, u in valid),
            "at_request_cap": sum((s.get("turns_used") or 0) >= 40 for s, _ in valid),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", type=Path)
    print(json.dumps(summarize(parser.parse_args().block), indent=2))


if __name__ == "__main__":
    main()
