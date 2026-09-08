"""Exact-token exports and conservative eligibility checks for Kokkos self-training."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypedDict

import tinker

from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
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


def _write_new_record(path: Path, payload: object) -> None:
    """Persist evidence before the next external action; never replace a turn."""
    encoded = json.dumps(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class RecordedTokenCompleter(TokenCompleter):
    """Keep raw sampling evidence even when the subsequent environment step fails."""

    def __init__(self, policy: TokenCompleter, directory: Path) -> None:
        self.policy = policy
        self.directory = directory
        self.calls_started = 0
        self.responses_received = 0

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        prefix = f"{self.calls_started:03d}"
        _write_new_record(
            self.directory / f"{prefix}.request.json",
            {"input_tokens": model_input.to_ints(), "stop": stop, "max_tokens": max_tokens},
        )
        self.calls_started += 1
        # Delegate unchanged, without adding a timeout or retry. A request
        # without a response remains an uncertain call, never a zero-cost slot.
        response = await self.policy(model_input, stop, max_tokens=max_tokens)
        self.responses_received += 1
        _write_new_record(
            self.directory / f"{prefix}.response.json",
            {
                "tokens": response.tokens,
                "logprobs": response.maybe_logprobs,
                "stop_reason": response.stop_reason,
            },
        )
        return response


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
        *,
        directory: Path | None = None,
    ) -> None:
        self.task = task
        self.sandbox = sandbox
        self.directory = directory if directory is not None else (
            results_dir / "rollouts" / f"{task.task_name}__{sample_index:02d}"
        )
        self.sample_index = sample_index
        self.reward_fn = reward_fn
        self.history: list[Message] = []
        self.patch = ""
        self.capture_error: str | None = None
        self.stage = "sampling"
        self.sampling: RecordedTokenCompleter | None = None

    def recording_policy(self, policy: TokenCompleter) -> RecordedTokenCompleter:
        self.sampling = RecordedTokenCompleter(policy, self.directory / "sampling")
        return self.sampling

    def save_failure(self, error: Exception) -> None:
        _write_new_record(
            self.directory / "failure.json",
            {
                "stage": self.stage,
                "error": f"{type(error).__name__}: {error}",
                "policy_calls_started": self.sampling.calls_started if self.sampling else 0,
                "responses_received": self.sampling.responses_received if self.sampling else 0,
                "trajectory_complete": False,
                "eligible_for_training": False,
            },
        )

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.history = history
        self.stage = "candidate_capture"
        _write_new_record(
            self.directory / "messages.json", [message_to_jsonable(m) for m in history]
        )
        # Do not stage files: intent-to-add turns ignored-by-the-verifier
        # untracked build outputs into tracked changes and can change rewards.
        self.directory.mkdir(parents=True, exist_ok=True)
        await capture_candidate(self.sandbox, task=self.task, results_dir=self.directory)
        metadata = json.loads((self.directory / "candidate.json").read_text())
        if not metadata["complete"]:
            self.capture_error = "patch_capture_failed_or_truncated"
        else:
            self.patch = (self.directory / "candidate.patch").read_text()
        self.stage = "grading"
        result = await self.reward_fn(history)
        self.stage = "graded"
        return result

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
        self.stage = "complete"
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
