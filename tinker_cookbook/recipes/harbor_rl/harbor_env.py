"""Harbor environment, dataset, and dataset builder for RL training."""

from __future__ import annotations

import logging
import random
import tomllib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborBashTool, HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.rollout_limits import RolloutLimits, TerminationRewardPolicy
from tinker_cookbook.rl.rollout_presets import RolloutConfig, agentic
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StopReason,
)
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn

logger = logging.getLogger(__name__)

HARBOR_CACHE_DIR = Path.home() / ".cache" / "harbor" / "tasks"
HARBOR_SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a sandboxed environment. "
    "You have access to a bash tool to execute commands. "
    "Complete the task described by the user."
)

SandboxFactory = Callable[[Path, int], Awaitable[SandboxInterface]]


async def default_sandbox_factory(
    env_dir: Path, timeout: int, *, allow_network: bool = True
) -> SandboxInterface:
    """Create a Modal sandbox from a task environment directory.

    Args:
        env_dir: Path to the task's environment/ directory (must contain a Dockerfile).
        timeout: Sandbox lifetime in seconds.
    """
    import modal

    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    dockerfile_path = env_dir / "Dockerfile"
    image = modal.Image.from_dockerfile(path=str(dockerfile_path), context_dir=str(env_dir))
    return await ModalSandbox.create(
        image=image,
        timeout=timeout,
        allow_network=allow_network,
    )


@dataclass(frozen=True)
class HarborTask:
    """A single Harbor terminal-bench task."""

    task_name: str
    instruction: str
    task_dir: Path  # Convention: environment/Dockerfile, tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)


def load_harbor_tasks(dataset: str) -> list[HarborTask]:
    """Load Harbor tasks from ~/.cache/harbor/tasks/<dataset>/."""
    return load_harbor_tasks_from_dir(HARBOR_CACHE_DIR / dataset)


def load_harbor_tasks_from_dir(tasks_dir: Path) -> list[HarborTask]:
    """Load Harbor tasks directly from a directory containing task folders."""
    tasks: list[HarborTask] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        tasks.append(
            HarborTask(
                task_name=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text(),
                task_dir=task_dir,
                config=tomllib.loads((task_dir / "task.toml").read_text()),
            )
        )
    tasks.sort(key=lambda t: t.task_name)
    return tasks


def _initial_messages(
    task: HarborTask,
    renderer: Renderer,
    bash_tool: HarborBashTool,
) -> list[Message]:
    """Build initial messages with tool schemas and task instruction."""
    tool_schemas = [bash_tool.bash.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=HARBOR_SYSTEM_PROMPT,
    )
    return prefix + [{"role": "user", "content": task.instruction}]


def harbor_termination_policy() -> TerminationRewardPolicy:
    """The grade-then-clamp policy Harbor RL should use, minus the turn cap.

    A Harbor task is graded by running ``test.sh`` against the repository state,
    so a trajectory that fixed the bug has earned its reward even if it then
    spent its last turn on one more read-only command. Clamping those to zero
    measured 22 of 71 verified successes away in a 20-step run -- and worse, fed
    them into their GRPO groups as zeros, making "fixed it but ran long"
    indistinguishable from "did not fix it".

    The remaining limit reasons keep the clamp: a token-exhausted or timed-out
    trajectory really is unfinished.

    This must be applied to **both** halves of the rollout config, which are
    resolved independently (see ``rl/train.py``'s note on ``Config.termination``):

    - trainer side: ``Config.termination``, consumed by
      ``Config.effective_termination()`` and passed to ``do_group_rollout``,
      which is where ``_apply_zero_reward_on_limit`` actually runs;
    - env side: ``build_agent_tool_env(rollout_config=...)``.

    Setting only the env side leaves the clamp active, because the trainer falls
    back to ``default_rollout_config_for_model``.
    """
    preset = agentic()
    return chz.replace(
        preset.termination,
        limit_stop_reasons=tuple(
            reason
            for reason in preset.termination.limit_stop_reasons
            if reason != StopReason.MAX_TURNS
        ),
    )


class HarborEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that creates Harbor environments with Modal sandboxes."""

    def __init__(
        self,
        task: HarborTask,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        sandbox_timeout: int = 600,
        command_timeout: int = 120,
        grader_timeout: int = 60,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        sandbox_factory: SandboxFactory | None = None,
        reward_fn: RewardFn | None = None,
        thinking_effort: float | None = None,
        raise_on_grading_error: bool = False,
        max_tool_calls: int | None = None,
    ):
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.grader_timeout = grader_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.sandbox_factory = sandbox_factory or default_sandbox_factory
        self.reward_fn = reward_fn
        self.thinking_effort = thinking_effort
        self.raise_on_grading_error = raise_on_grading_error
        self.max_tool_calls = max_tool_calls
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []

        env_dir = self.task.task_dir / "environment"

        # Create renderer (stateless, shared across envs)
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        tests_dir = self.task.task_dir / "tests"

        envs = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(env_dir, self.sandbox_timeout)
            self._sandboxes.append(sandbox)

            bash_tool = HarborBashTool(sandbox, command_timeout=self.command_timeout)
            reward_fn = self.reward_fn or HarborReward(
                tests_dir=tests_dir,
                sandbox=sandbox,
                grader_timeout=self.grader_timeout,
                raise_on_grading_error=self.raise_on_grading_error,
                task_name=self.task.task_name,
            )
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[bash_tool.bash],
                    initial_messages=_initial_messages(self.task, renderer, bash_tool),
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                    model_name=self.model_name,
                    rollout_config=self._rollout_config(),
                    generation_prompt_kwargs=(
                        {"effort": self.thinking_effort}
                        if self.thinking_effort is not None
                        else None
                    ),
                )
            )
        return envs

    def _rollout_config(self) -> RolloutConfig:
        """Build a rollout config carrying this env's own budgets.

        Without this, ``build_agent_tool_env`` falls back to
        ``default_rollout_config_for_model``, and the runner enforces that preset's
        ``RolloutLimits`` (for Inkling: 10 turns, 30 tool calls, 64K trajectory
        tokens) in preference to the limits configured here -- silently capping
        agentic rollouts far below the requested budget. ``eval.py`` builds the
        same config for the same reason, so training and evaluation stay comparable.

        The termination policy comes from :func:`harbor_termination_policy`; note
        that this only covers the env-side half, and the trainer side must set
        ``Config.termination`` to the same value.
        """
        preset = agentic()
        return RolloutConfig(
            limits=RolloutLimits(
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tool_calls=self.max_tool_calls,
            ),
            parse_errors=preset.parse_errors,
            termination=harbor_termination_policy(),
            tool_execution=preset.tool_execution,
        )

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            try:
                await sandbox.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed: %s", e)
        self._sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["harbor"]


class HarborDataset(RLDataset):
    """Dataset that produces batches of HarborEnvGroupBuilders."""

    def __init__(
        self,
        env_group_builders: list[HarborEnvGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    """Build an RL dataset over Harbor tasks."""

    tasks: list[HarborTask]
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str | None = None
    max_turns: int = 10
    sandbox_timeout: int = 600
    command_timeout: int = 120
    grader_timeout: int = 60
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    sandbox_factory: SandboxFactory | None = None
    reward_fn: RewardFn | None = None
    thinking_effort: float | None = None
    raise_on_grading_error: bool = False
    # Cap on total tool calls per rollout; None leaves it to the turn limit.
    max_tool_calls: int | None = None

    # Held-out split. ``eval_size=0`` keeps the legacy behaviour of evaluating on
    # every task (in-sample). With ``eval_size > 0`` the tasks are shuffled with
    # ``seed`` and split disjointly, and the builder returns ``None`` for the test
    # slot so the auto-wrapped evaluator in ``rl.train`` does not also spin up an
    # in-sample eval over every task; use a dedicated held-out evaluator instead.
    eval_size: int = 0
    eval_group_size: int = 1
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Summarise for config dumps instead of inlining every task.

        ``ml_log.dump_config`` prefers ``to_dict``. Without it, each task's full
        instruction text and TOML config is inlined into the run config -- ~200KB
        for a 100-task dataset, which experiment trackers reject. Task identity is
        preserved via names, and the split is reproducible from ``seed``.
        """
        train_tasks, held_out_tasks = self.split_tasks()
        return {
            "num_tasks": len(self.tasks),
            "train_task_names": [task.task_name for task in train_tasks],
            "held_out_task_names": [task.task_name for task in held_out_tasks],
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "model_name": self.model_name,
            "renderer_name": self.renderer_name,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "sandbox_timeout": self.sandbox_timeout,
            "command_timeout": self.command_timeout,
            "grader_timeout": self.grader_timeout,
            "max_trajectory_tokens": self.max_trajectory_tokens,
            "max_generation_tokens": self.max_generation_tokens,
            "context_overflow_reward": self.context_overflow_reward,
            "thinking_effort": self.thinking_effort,
            "raise_on_grading_error": self.raise_on_grading_error,
            "eval_size": self.eval_size,
            "eval_group_size": self.eval_group_size,
            "seed": self.seed,
            "sandbox_factory": type(self.sandbox_factory).__name__
            if self.sandbox_factory is not None
            else None,
        }

    def _make_env_group_builders(
        self, group_size: int, tasks: list[HarborTask] | None = None
    ) -> list[HarborEnvGroupBuilder]:
        return [
            HarborEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                sandbox_factory=self.sandbox_factory,
                reward_fn=self.reward_fn,
                thinking_effort=self.thinking_effort,
                raise_on_grading_error=self.raise_on_grading_error,
                max_tool_calls=self.max_tool_calls,
            )
            for task in (self.tasks if tasks is None else tasks)
        ]

    def split_tasks(self) -> tuple[list[HarborTask], list[HarborTask]]:
        """Split tasks into ``(train, held_out)`` deterministically.

        Tasks arrive sorted by ``task_name`` from :func:`load_harbor_tasks_from_dir`,
        so shuffling with a fixed ``seed`` is reproducible. Returns an empty
        held-out list when ``eval_size`` is 0.
        """
        if self.eval_size == 0:
            return list(self.tasks), []
        if not 0 <= self.eval_size < len(self.tasks):
            raise ValueError(
                f"eval_size must be in [0, {len(self.tasks) - 1}], got {self.eval_size}"
            )
        shuffled = list(self.tasks)
        random.Random(self.seed).shuffle(shuffled)
        return shuffled[self.eval_size :], shuffled[: self.eval_size]

    def make_held_out_env_group_builders(self) -> list[HarborEnvGroupBuilder]:
        """Build one env group builder per held-out task, at ``eval_group_size``.

        Returns an empty list when no held-out split is configured. Use this to
        construct a dedicated held-out evaluator, since ``__call__`` deliberately
        returns ``None`` for the test slot when ``eval_size > 0``.
        """
        _, held_out_tasks = self.split_tasks()
        if not held_out_tasks:
            return []
        return self._make_env_group_builders(self.eval_group_size, held_out_tasks)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_tasks, held_out_tasks = self.split_tasks()
        train_dataset = HarborDataset(
            env_group_builders=self._make_env_group_builders(self.group_size, train_tasks),
            batch_size=self.batch_size,
        )
        if held_out_tasks:
            # A held-out split is measured by a dedicated evaluator that bounds
            # sandbox concurrency; returning None keeps rl.train from also
            # auto-wrapping an in-sample eval over the training tasks.
            logger.info(
                "Harbor split: %d train tasks, %d held out (%s)",
                len(train_tasks),
                len(held_out_tasks),
                ", ".join(task.task_name for task in held_out_tasks),
            )
            return train_dataset, None
        eval_dataset = HarborDataset(
            env_group_builders=self._make_env_group_builders(group_size=1),
            batch_size=self.batch_size,
        )
        return train_dataset, eval_dataset
