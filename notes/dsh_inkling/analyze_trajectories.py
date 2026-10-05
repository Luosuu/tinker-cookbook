"""Failure-mode analysis for Inkling-Small Kokkos trajectories.

Reads a trajectory corpus in the format produced by the Kokkos self-training
pipeline (one JSON list, each element a rollout with ``messages``,
``instruction``, ``patch``, ``logical_score``, ``stop_reason`` and
``sampled_tokens``) and reports where episodes lose score.

The point of the analysis is to choose harness changes for DSH rather than to
produce a benchmark number. Every reported statistic is therefore tied to a
mechanism that a harness edit could plausibly move.

Usage::

    python -m notes.dsh_inkling.analyze_trajectories \
        notes/genesis_archive/round01/trajectories.json \
        --baseline-checkpoint thinkingmachines/Inkling-Small:peft:262144
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# A write is any command that can modify the repository. Deliberately broad in
# what counts as an edit, but redirects to /dev/null and fd duplication are
# stripped first: `grep -rn foo . 2>/dev/null` is a read, not a write, and
# treating it as one would invent exploration-free episodes that never existed.
_DEVNULL_REDIRECT = re.compile(r"\d*&?>>?\s*/dev/null|\d*>&\s*\d+|&>>?\s*/dev/null")

_WRITE_PATTERN = re.compile(
    r"cat\s*>|tee\s|>>?\s*\S|sed\s+-i|patch\s+-|git\s+apply|applypatch"
    r"|python[0-9.]*\s+-\s*<<|<<\s*'?EOF|python[0-9.]*\s+-c\s+.*(write|open\()",
)


def _is_write(command: str) -> bool:
    """True when the command plausibly modifies files on disk."""
    if not command:
        return False
    without_devnull = _DEVNULL_REDIRECT.sub(" ", command)
    return bool(_WRITE_PATTERN.search(without_devnull))


_BUILD_PATTERN = re.compile(
    r"\b(cmake|make|ninja|g\+\+|gcc|clang|ctest)\b",
)

_READ_PATTERN = re.compile(
    r"^\s*(grep|rg|find|ls|cat|sed|head|tail|wc|awk|less|file|stat|diff|echo\s+-n)\b",
)


@dataclass
class Episode:
    """One rollout, reduced to the harness-relevant fields."""

    task: str
    sample: int
    stop_reason: str | None
    score: float | None
    turns_used: int | None
    sampled_tokens: int | None
    audit_flags: list[str]
    has_patch: bool
    commands: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)

    @property
    def empty_patch(self) -> bool:
        return not self.has_patch

    def turn_of_first_write(self) -> int | None:
        """Index of the assistant turn whose tool call first wrote to disk.

        ``None`` means the episode never attempted a write.
        """
        turn = 0
        for message in self._raw_messages:
            for call in message.get("tool_calls") or []:
                arguments = _tool_arguments(call)
                if _is_write(arguments.get("command") or ""):
                    return turn
            if message.get("role") == "assistant":
                turn += 1
        return None

    _raw_messages: list[dict] = field(default_factory=list, repr=False)


def _tool_arguments(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _classify(command: str) -> str:
    stripped = command.strip()
    if _BUILD_PATTERN.search(stripped):
        return "build"
    if _READ_PATTERN.search(stripped):
        return "read/search"
    if re.match(r"^\s*(git|python|pip|apt|uv)\b", stripped):
        return "tooling"
    return "other"


def load_episodes(path: Path, checkpoint_prefix: str | None) -> list[Episode]:
    raw = json.loads(path.read_text())
    episodes: list[Episode] = []
    for entry in raw:
        checkpoint = entry.get("checkpoint") or ""
        if checkpoint_prefix and not checkpoint.startswith(checkpoint_prefix):
            continue
        messages = entry.get("messages") or []
        commands: list[str] = []
        thinking: list[str] = []
        for message in messages:
            for call in message.get("tool_calls") or []:
                command = _tool_arguments(call).get("command")
                if command:
                    commands.append(command)
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking.append(block.get("thinking") or "")
        result = entry.get("result") or {}
        episodes.append(
            Episode(
                task=entry.get("task") or "?",
                sample=entry.get("sample") or 0,
                stop_reason=entry.get("stop_reason"),
                score=entry.get("logical_score"),
                turns_used=result.get("turns_used"),
                sampled_tokens=entry.get("sampled_tokens"),
                audit_flags=list(entry.get("audit_flags") or []),
                has_patch=bool((entry.get("patch") or "").strip()),
                commands=commands,
                thinking=thinking,
                _raw_messages=messages,
            )
        )
    return episodes


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def report_stop_reasons(episodes: list[Episode]) -> None:
    print("== Outcome by stop reason ==")
    print(
        f"{'stop_reason':14} {'n':>5} {'share':>7} {'pass':>7} {'mean_turns':>11} {'mean_tokens':>12}"
    )
    for reason in ("completed", "max_turns", "parse_error", None, "timeout"):
        rows = [e for e in episodes if e.stop_reason == reason]
        if not rows:
            continue
        passes = sum(1 for e in rows if e.score == 1.0)
        turns = [e.turns_used for e in rows if e.turns_used]
        tokens = [e.sampled_tokens for e in rows if e.sampled_tokens]
        mean_turns = sum(turns) / len(turns) if turns else 0.0
        mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0
        print(
            f"{reason!s:14} {len(rows):5} {_pct(len(rows), len(episodes)):>7} "
            f"{_pct(passes, len(rows)):>7} {mean_turns:11.1f} {mean_tokens:12.0f}"
        )


def report_score_decomposition(episodes: list[Episode]) -> None:
    """Show how much of the total score each termination class carries.

    A class that contributes passes *and* holds most of the corpus is the one
    worth editing the harness for.
    """
    print("\n== Where the score comes from ==")
    total = len(episodes)
    passes = sum(1 for e in episodes if e.score == 1.0)
    print(f"episodes={total}  passed={passes}  score={_pct(passes, total)}")
    buckets: collections.Counter[tuple[str, str]] = collections.Counter()
    for episode in episodes:
        outcome = "PASS" if episode.score == 1.0 else "fail"
        buckets[(outcome, str(episode.stop_reason))] += 1
    for (outcome, reason), count in sorted(buckets.items()):
        print(f"  {outcome:5} {reason:14} n={count:5} ({_pct(count, total)})")

    capped = [e for e in episodes if e.stop_reason == "max_turns"]
    no_patch = [e for e in capped if e.empty_patch]
    print(
        f"\n  capped with a patch:    {len(capped) - len(no_patch):4} "
        f"(pass {_pct(sum(1 for e in capped if not e.empty_patch and e.score == 1.0), max(len(capped) - len(no_patch), 1))})"
    )
    print(
        f"  capped with NO patch:   {len(no_patch):4} "
        f"(pass {_pct(sum(1 for e in no_patch if e.score == 1.0), max(len(no_patch), 1))})"
    )


def report_exploration(episodes: list[Episode]) -> None:
    """Measure the read/act split, the core harness hypothesis.

    If most episodes spend their budget reading and reach a write late or never,
    the harness lacks a forcing function that converts evidence into an edit.
    """
    print("\n== Read/act behaviour ==")
    completed = [e for e in episodes if e.stop_reason == "completed"]
    capped = [e for e in episodes if e.stop_reason == "max_turns"]

    for label, rows in (("completed", completed), ("max_turns", capped)):
        if not rows:
            continue
        wrote = [e for e in rows if e.turn_of_first_write() is not None]
        firsts = [e.turn_of_first_write() for e in wrote]
        mean_first = sum(t for t in firsts if t is not None) / len(firsts) if firsts else 0.0
        print(
            f"{label:11} n={len(rows):4}  ever_wrote={_pct(len(wrote), len(rows)):>6}  "
            f"mean_turn_of_first_write={mean_first:5.1f}"
        )

    kinds: collections.Counter[str] = collections.Counter()
    for episode in episodes:
        for command in episode.commands:
            kinds[_classify(command)] += 1
    total_commands = sum(kinds.values())
    if total_commands:
        print(f"\ncommand mix over {total_commands} calls:")
        for kind, count in kinds.most_common():
            print(f"  {kind:12} {count:5} ({_pct(count, total_commands)})")


def report_repetition(episodes: list[Episode]) -> None:
    """Check whether capped episodes are looping rather than working.

    Looping and slow-but-progressing failures call for opposite harness fixes:
    a repeat detector versus a stronger summary or a shorter exploration phase.
    """
    capped = [e for e in episodes if e.stop_reason == "max_turns" and e.empty_patch]
    ratios: list[tuple[float, Episode, str]] = []
    for episode in capped:
        if not episode.commands:
            continue
        counts = collections.Counter(episode.commands)
        _, top = counts.most_common(1)[0]
        ratios.append((top / len(episode.commands), episode, counts.most_common(1)[0][0]))
    if not ratios:
        print("\n== Repetition ==\nno capped empty-patch episodes")
        return
    mean_ratio = sum(r for r, _, _ in ratios) / len(ratios)
    looping = sum(1 for r, _, _ in ratios if r > 0.5)
    print("\n== Repetition in capped, empty-patch episodes ==")
    print(
        f"n={len(ratios)}  mean_top_command_share={mean_ratio:.3f}  episodes >50% repeats={looping}"
    )
    print("most repetitive:")
    ratios.sort(key=lambda item: item[0], reverse=True)
    for ratio, episode, command in ratios[:5]:
        print(f"  {ratio:.2f} {episode.task:26} {command[:80]}")


def report_audit_flags(episodes: list[Episode]) -> None:
    print("\n== Audit flags ==")
    counts = collections.Counter(",".join(e.audit_flags) or "none" for e in episodes)
    for flags, count in counts.most_common():
        print(f"  {flags:40} {count:5} ({_pct(count, len(episodes))})")


def report_tasks(episodes: list[Episode]) -> None:
    """Separate unreachable tasks from tasks the harness is losing."""
    per_task: dict[str, list[Episode]] = collections.defaultdict(list)
    for episode in episodes:
        per_task[episode.task].append(episode)
    scored = []
    for task, rows in per_task.items():
        scored.append((sum(1 for e in rows if e.score == 1.0) / len(rows), task, len(rows)))
    scored.sort()
    print(f"\n== Per-task difficulty (n_tasks={len(scored)}) ==")
    print("always failing (candidate harness or task-capability limits):")
    for rate, task, n in scored:
        if rate == 0.0:
            print(f"  {rate:.2f} n={n} {task}")
    always_passing = [s for s in scored if s[0] == 1.0]
    print(f"always passing: {len(always_passing)} tasks")
    print(f"mixed (harness headroom): {sum(1 for s in scored if 0.0 < s[0] < 1.0)} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="trajectories.json to analyze")
    parser.add_argument(
        "--baseline-checkpoint",
        default="thinkingmachines/Inkling-Small",
        help="only include episodes whose checkpoint starts with this prefix",
    )
    parser.add_argument("--tasks", action="store_true", help="also print the per-task breakdown")
    args = parser.parse_args()

    episodes = load_episodes(args.corpus, args.baseline_checkpoint)
    if not episodes:
        raise SystemExit(f"no episodes matched in {args.corpus}")
    print(
        f"corpus={args.corpus}  episodes={len(episodes)}  checkpoint~={args.baseline_checkpoint}\n"
    )
    report_stop_reasons(episodes)
    report_score_decomposition(episodes)
    report_exploration(episodes)
    report_repetition(episodes)
    report_audit_flags(episodes)
    if args.tasks:
        report_tasks(episodes)


if __name__ == "__main__":
    main()
