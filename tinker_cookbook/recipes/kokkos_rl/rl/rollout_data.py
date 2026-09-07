"""Exact-token exports and conservative eligibility checks for Kokkos self-training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
from tinker_cookbook.renderers.base import Message, message_to_jsonable
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.sandbox import SandboxInterface


class ExportedDatum(TypedDict):
    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]


class RecordedRollout(TypedDict):
    task_name: str
    sample_index: int
    reward: float
    turns: int
    sampled_tokens: int
    stop_reason: str | None
    audit_flags: list[str]
    datums: list[ExportedDatum]


def command_audit(history: list[Message]) -> list[str]:
    """Flag answer lookup and verifier access; a fresh patch regrade is also required."""
    flags: set[str] = set()
    for message in history:
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls", []):
            arguments = call.function.arguments
            if re.search(
                r"github\.com|api\.github|git\s+(?:show|log|cherry-pick|fetch|pull)", arguments
            ):
                flags.add("answer_lookup")
            if re.search(
                r"(?<![\w./-])/(?:tests|solution)(?:/|(?=[\s\"'\\]|$))"
                r"|/logs/verifier|test\.patch|gold\.patch",
                arguments,
            ):
                flags.add("verifier_access")
    return sorted(flags)


class RolloutRecorder:
    """Capture source edits before the grader injects tests, without changing prompts."""

    def __init__(
        self,
        task: HarborTask,
        sandbox: SandboxInterface,
        results_dir: Path,
        sample_index: int,
        reward_fn: HarborReward,
    ) -> None:
        self.task = task
        self.sandbox = sandbox
        self.directory = results_dir / "rollouts" / f"{task.task_name}__{sample_index:02d}"
        self.sample_index = sample_index
        self.reward_fn = reward_fn
        self.history: list[Message] = []
        self.patch = ""
        self.capture_error: str | None = None

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.history = history
        # Do not stage files: intent-to-add turns ignored-by-the-verifier
        # untracked build outputs into tracked changes and can change rewards.
        self.directory.mkdir(parents=True, exist_ok=True)
        await capture_candidate(self.sandbox, task=self.task, results_dir=self.directory)
        metadata = json.loads((self.directory / "candidate.json").read_text())
        if not metadata["complete"]:
            self.capture_error = "patch_capture_failed_or_truncated"
        else:
            self.patch = (self.directory / "candidate.patch").read_text()
        return await self.reward_fn(history)

    def save(self, trajectory: Trajectory) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        flags = command_audit(self.history)
        if self.capture_error:
            flags.append(self.capture_error)
        if not self.patch:
            flags.append("empty_patch")
        if any(
            t.metrics.get("parse_error") or t.metrics.get("parse_error_masked")
            for t in trajectory.transitions
        ):
            flags.append("parse_error")
        datums = trajectory_to_data(trajectory, 1.0)
        payload = {
            "task_name": self.task.task_name,
            "sample_index": self.sample_index,
            "reward": sum(t.reward for t in trajectory.transitions),
            "turns": len(trajectory.transitions),
            "sampled_tokens": sum(len(t.ac.tokens) for t in trajectory.transitions),
            "stop_reason": trajectory.stop_reason,
            "audit_flags": flags,
            "datums": [
                {
                    "input_tokens": d.model_input.to_ints(),
                    "target_tokens": list(d.loss_fn_inputs["target_tokens"].data),
                    "weights": list(d.loss_fn_inputs["mask"].data),
                }
                for d in datums
            ],
        }
        (self.directory / "patch.diff").write_text(self.patch)
        messages = [message_to_jsonable(m) for m in self.history]
        (self.directory / "messages.json").write_text(json.dumps(messages))
        (self.directory / "trajectory.json").write_text(json.dumps(payload))


def eligible(row: RecordedRollout, max_turns: int = 40) -> bool:
    """Only naturally completed, unflagged successes are demonstration candidates.

    A passing cap-terminated episode still counts in evaluation, but does not
    teach a natural stopping action and is excluded from both SFT arms.
    """
    return (
        row["reward"] == 1
        and 0 < row["turns"] <= max_turns
        and row["stop_reason"] == "completed"
        and not row["audit_flags"]
        and bool(row["datums"])
    )
