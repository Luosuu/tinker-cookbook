"""Failure taxonomy for a Rivanna block, read from each episode's own evidence.

Reads native, DSH and Pi episodes in the ``runs/<arm>-d<draw>/results/<task>/`` layout and
classifies every validly graded failure by what the verifier saw (no source change, failed
build, failed fail-to-pass test, failed pass-to-pass test) and by how the episode ended.
Only within-task comparisons support a lever: aggregate correlations reversed under task
control in the DSH study, so the per-task section is the one to read first.

Usage::

    python -m notes.pi_harness.taxonomy BLOCK [--arms native,pi]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WRITE = re.compile(
    r"cat\s*>|tee\s|sed\s+-i|patch\s+-|git\s+apply|>>?\s*/workspace|<<\s*'?[A-Z]+'?|"
    r"open\([^)]*['\"]w"
)
BUILD = re.compile(r"\b(cmake\s+--build|ninja|ctest|make)\b")
# Untracked files the sandbox itself writes; they are not candidate source changes.
NOT_SOURCE = re.compile(r"^(build/|workspace/|\.|logs/)")


@dataclass
class Episode:
    arm: str
    draw: str
    task: str
    reward: float
    error: str | None
    turns: int
    commands: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        if self.error:
            return "infra_error"
        if self.reward == 1.0:
            return "solved"
        if not self.source_files:
            return "no_source_change"
        failed = [c for c in self.checks if c.get("status") == "failed"]
        if not failed:
            return "failed_unclassified"
        first = failed[0]
        if first.get("stage") == "build":
            return "build_failed"
        ident = str(first.get("id", ""))
        if ident.startswith("p2p") or first.get("role") == "p2p":
            return "p2p_failed"
        if ident.startswith("f2p") or first.get("role") == "f2p":
            return "f2p_failed"
        return f"{first.get('stage', 'other')}_failed"

    @property
    def last_edit_unbuilt(self) -> bool:
        last_edit = max((i for i, c in enumerate(self.commands) if WRITE.search(c)), default=-1)
        last_build = max((i for i, c in enumerate(self.commands) if BUILD.search(c)), default=-1)
        return last_edit >= 0 and last_build < last_edit


def _commands(directory: Path) -> list[str]:
    events = directory / "events.jsonl"
    if events.is_file():
        out = []
        for line in events.read_text().splitlines():
            event = json.loads(line)
            # Harness build checks and state probes are not the model's commands.
            if (
                event.get("type") == "bash"
                and not event.get("internal")
                and isinstance(event.get("command"), str)
            ):
                out.append(event["command"])
        return out
    out = []
    for transcript in directory.glob("kokkos__*.json"):
        for turn in json.loads(transcript.read_text()):
            for call in turn.get("function_calls", []):
                try:
                    out.append(json.loads(call["arguments"])["command"])
                except (ValueError, KeyError, TypeError):
                    continue
    return out


def _checks(directory: Path) -> list[dict]:
    for path in [directory / "grading.json", *directory.glob("grading_logs/*.json")]:
        if path.is_file():
            diagnostics = json.loads(path.read_text()).get("diagnostics") or {}
            return list(diagnostics.get("checks") or [])
    return []


def _source_files(directory: Path) -> list[str]:
    patch = directory / "candidate.patch"
    if not patch.is_file():
        return []
    files = re.findall(r"^diff --git a/(\S+) b/", patch.read_text(errors="replace"), re.MULTILINE)
    return [f for f in files if not NOT_SOURCE.match(f)]


def load_block(block: Path, arms: list[str] | None = None) -> list[Episode]:
    episodes = []
    for summary_path in sorted(block.glob("runs/*/results/*/summary.json")):
        directory = summary_path.parent
        arm, draw = directory.parents[1].name.rsplit("-d", 1)
        if arms and arm not in arms:
            continue
        summary = json.loads(summary_path.read_text())
        episodes.append(
            Episode(
                arm=arm,
                draw=draw,
                task=directory.name,
                reward=float(summary.get("reward") or 0.0),
                error=summary.get("error"),
                turns=int(summary.get("turns_used") or 0),
                commands=_commands(directory),
                checks=_checks(directory),
                source_files=_source_files(directory),
            )
        )
    return episodes


def report(episodes: list[Episode]) -> dict:
    by_arm: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_arm[episode.arm].append(episode)
    out: dict = {"arms": {}}
    for arm, items in sorted(by_arm.items()):
        valid = [e for e in items if not e.error]
        failed = [e for e in valid if e.reward != 1.0]
        outcomes = Counter(e.outcome for e in failed)
        out["arms"][arm] = {
            "episodes": len(items),
            "valid": len(valid),
            "solved": sum(e.reward == 1.0 for e in valid),
            "failure_outcomes": dict(outcomes.most_common()),
            "failed_at_request_cap": sum(e.turns >= 40 for e in failed),
            "solved_at_request_cap": sum(e.turns >= 40 for e in valid if e.reward == 1.0),
            "failed_with_unbuilt_final_edit": sum(e.last_edit_unbuilt for e in failed),
            "median_turns_failed": sorted(e.turns for e in failed)[len(failed) // 2]
            if failed
            else None,
        }
    per_task: dict[str, Counter] = defaultdict(Counter)
    for e in episodes:
        if not e.error:
            per_task[e.task]["n"] += 1
            per_task[e.task]["solved"] += e.reward == 1.0
    groups = Counter()
    for counts in per_task.values():
        rate = counts["solved"] / counts["n"]
        groups["never" if rate == 0 else "always" if rate == 1 else "mixed"] += 1
    out["tasks"] = dict(groups)
    out["mixed_tasks"] = sorted(
        (t for t, c in per_task.items() if 0 < c["solved"] < c["n"]),
        key=lambda t: per_task[t]["solved"] / per_task[t]["n"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", type=Path)
    parser.add_argument("--arms")
    args = parser.parse_args()
    arms = args.arms.split(",") if args.arms else None
    print(json.dumps(report(load_block(args.block, arms)), indent=2))


if __name__ == "__main__":
    main()
