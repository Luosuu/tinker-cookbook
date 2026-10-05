"""Analyze a DSH-on-Rivanna run directory in the terms the reference analysis uses.

Each episode leaves `events.jsonl` (sandbox commands), `proxy-budget.jsonl` (requests,
tokens, message digests) and `summary.json` (outcome). This turns a run directory into the
same questions asked of the self-training corpus: where does the budget go, does the agent
ever edit, and how much of the run is still exploring when it stops.

Usage::

    python -m notes.dsh_inkling.analyze_run <run-dir> [--reference <progress.json>]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from notes.dsh_inkling.analyze_trajectories import _is_write

# Commands that only inspect the repository.
_READ = re.compile(r"^\s*(grep|rg|find|ls|cat|sed|head|tail|wc|awk|less|file|stat|diff)\b")
_BUILD = re.compile(r"\b(cmake|ninja|make|g\+\+|gcc|clang|ctest)\b")


@dataclass
class Episode:
    task: str
    reward: float | None
    status: str | None
    error: str | None
    turns: int | None
    seconds: float | None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    commands: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    system_chars: int | None = None

    def turn_of_first_write(self) -> int | None:
        for index, command in enumerate(self.commands):
            if _is_write(command):
                return index
        return None

    @property
    def wrote(self) -> bool:
        return self.turn_of_first_write() is not None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_run(run_dir: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for result_dir in sorted((run_dir / "results").glob("*")):
        if not result_dir.is_dir():
            continue
        summary = _read_json(result_dir / "summary.json") or {}
        episode = Episode(
            task=summary.get("task_name") or result_dir.name,
            reward=summary.get("reward"),
            status=summary.get("harness_status"),
            error=summary.get("error"),
            turns=summary.get("turns_used"),
            seconds=summary.get("time_seconds"),
        )
        events_path = result_dir / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "bash" and event.get("command"):
                    episode.commands.append(event["command"])
        budget_path = result_dir / "proxy-budget.jsonl"
        if budget_path.exists():
            first = True
            for line in budget_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") == "usage":
                    snapshot = record.get("budget") or {}
                    episode.requests = snapshot.get("requests", episode.requests)
                    episode.input_tokens = snapshot.get("input_tokens", episode.input_tokens)
                    episode.output_tokens = snapshot.get("output_tokens", episode.output_tokens)
                elif record.get("event") == "request":
                    snapshot = record.get("budget") or {}
                    episode.requests = snapshot.get("requests", episode.requests)
                    # A request snapshot also carries the running totals, so adopt them:
                    # attribution must not depend on whether a usage frame was captured.
                    episode.input_tokens = snapshot.get("input_tokens", episode.input_tokens)
                    episode.output_tokens = snapshot.get("output_tokens", episode.output_tokens)
                    if first:
                        episode.tools = list(record.get("tools") or [])
                        episode.roles = [m.get("role") for m in record.get("messages") or []]
                        episode.system_chars = (record.get("messages") or [{}])[0].get("chars")
                        first = False
        episodes.append(episode)
    return episodes


def report_prompt_parity(episodes: list[Episode]) -> bool:
    """Every episode must carry the reference prompt shape or its score is not comparable."""
    print("== Prompt parity ==")
    offenders = []
    for episode in episodes:
        if episode.roles != ["system", "user"] or episode.tools != ["bash"]:
            offenders.append(episode)
    if not offenders:
        sizes = {e.system_chars for e in episodes if e.system_chars is not None}
        print(
            f"  all {len(episodes)} episodes: roles=['system','user'] tools=['bash'] "
            f"system_chars={sorted(sizes)}"
        )
        return True
    print(f"  {len(offenders)}/{len(episodes)} episodes diverge:")
    for episode in offenders[:5]:
        print(f"    {episode.task}: roles={episode.roles} tools={episode.tools}")
    return False


def report_outcomes(episodes: list[Episode]) -> None:
    print("\n== Outcomes ==")
    total = len(episodes)
    scored = [e for e in episodes if e.reward is not None]
    solved = sum(1 for e in scored if e.reward == 1.0)
    print(
        f"  episodes={total} scored={len(scored)} solved={solved} pass@1={solved / len(scored):.1%}"
        if scored
        else "  no scored episodes"
    )
    statuses = collections.Counter(e.status or "unknown" for e in episodes)
    for status, count in statuses.most_common():
        rows = [e for e in episodes if (e.status or "unknown") == status]
        passed = sum(1 for e in rows if e.reward == 1.0)
        print(f"  {status:12} n={count:4} pass={passed / count:.1%}")
    errors = [e for e in episodes if e.error]
    print(f"  episodes with an error field: {len(errors)}")


def report_budget(episodes: list[Episode]) -> None:
    print("\n== Budget use ==")
    capped = [e for e in episodes if e.status and e.status.startswith("exit")]
    finished = [e for e in episodes if e.status == "finished"]
    for label, rows in (("finished", finished), ("capped", capped)):
        if not rows:
            continue
        turns = [e.turns for e in rows if e.turns is not None]
        inputs = [e.input_tokens for e in rows if e.input_tokens]
        outputs = [e.output_tokens for e in rows if e.output_tokens]
        mean_turns = sum(turns) / len(turns) if turns else 0.0
        mean_in = sum(inputs) / len(inputs) if inputs else 0.0
        mean_out = sum(outputs) / len(outputs) if outputs else 0.0
        passed = sum(1 for e in rows if e.reward == 1.0)
        print(
            f"  {label:9} n={len(rows):4} pass={passed / len(rows):6.1%} "
            f"mean_requests={mean_turns:5.1f} mean_in={mean_in:9.0f} mean_out={mean_out:7.0f}"
        )
    zero_inputs = sum(1 for e in episodes if e.requests and not e.input_tokens)
    if zero_inputs:
        print(
            f"  WARNING: {zero_inputs} episodes recorded requests but no input tokens; "
            "token accounting is broken"
        )


def report_behavior(episodes: list[Episode]) -> None:
    print("\n== Read/act behaviour ==")
    for label, rows in (
        ("solved", [e for e in episodes if e.reward == 1.0]),
        ("failed", [e for e in episodes if e.reward == 0.0]),
    ):
        if not rows:
            continue
        wrote = [e for e in rows if e.wrote]
        firsts = [e.turn_of_first_write() for e in wrote]
        mean_first = sum(t for t in firsts if t is not None) / len(firsts) if firsts else 0.0
        print(
            f"  {label:7} n={len(rows):4} ever_wrote={len(wrote) / len(rows):6.1%} "
            f"mean_first_write_call={mean_first:5.1f}"
        )

    kinds: collections.Counter[str] = collections.Counter()
    for episode in episodes:
        for command in episode.commands:
            if _BUILD.search(command):
                kinds["build"] += 1
            elif _READ.match(command):
                kinds["read/search"] += 1
            else:
                kinds["other"] += 1
    total = sum(kinds.values())
    if total:
        print(f"\n  command mix over {total} calls:")
        for kind, count in kinds.most_common():
            print(f"    {kind:12} {count:6} ({count / total:.1%})")

    capped_no_write = [
        e for e in episodes if e.status and e.status.startswith("exit") and not e.wrote
    ]
    print(
        f"\n  capped with no write attempt: {len(capped_no_write)}/{len(episodes)} "
        f"({len(capped_no_write) / len(episodes):.1%})"
    )


def report_vs_reference(episodes: list[Episode], reference: Path) -> None:
    """Compare per-task against the reference run, and name what the harness wins/loses."""
    try:
        progress = json.loads(reference.read_text())
    except (OSError, ValueError):
        print(f"\n== vs reference ==\n  could not read {reference}")
        return
    reference_rewards = {
        row["task"]: row["reward"]
        for row in progress.get("trials", [])
        if row.get("model", "").endswith("Inkling-Small:peft:262144")
    }
    if not reference_rewards:
        print("\n== vs reference ==\n  no Inkling-Small trials in the reference file")
        return
    paired = [e for e in episodes if e.reward is not None and e.task in reference_rewards]
    if not paired:
        print("\n== vs reference ==\n  no overlapping tasks")
        return
    wins = [e for e in paired if e.reward == 1.0 and reference_rewards[e.task] == 0.0]
    losses = [e for e in paired if e.reward == 0.0 and reference_rewards[e.task] == 1.0]
    dsh = sum(1 for e in paired if e.reward == 1.0)
    ref = sum(1 for e in paired if reference_rewards[e.task] == 1.0)
    print(f"\n== vs reference (n={len(paired)}) ==")
    print(
        f"  DSH={dsh}/{len(paired)} ({dsh / len(paired):.1%})  "
        f"reference={ref}/{len(paired)} ({ref / len(paired):.1%})  "
        f"delta={dsh - ref:+d}"
    )
    print(f"  gained ({len(wins)}): {' '.join(e.task for e in wins) or '-'}")
    print(f"  lost   ({len(losses)}): {' '.join(e.task for e in losses) or '-'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()

    episodes = load_run(args.run_dir)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.run_dir}")
    print(f"run={args.run_dir}  episodes={len(episodes)}\n")
    report_outcomes(episodes)
    report_prompt_parity(episodes)
    report_budget(episodes)
    report_behavior(episodes)
    if args.reference:
        report_vs_reference(episodes, args.reference)


if __name__ == "__main__":
    main()
