import asyncio
import json
from pathlib import Path

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDatasetBuilder, HarborTask
from tinker_cookbook.recipes.harbor_rl.heldout_eval import (
    BoundedConcurrencyEvaluator,
    BoundedHeldOutEvaluatorBuilder,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

MODEL = "thinkingmachines/Inkling-Small:peft:262144"


def _tasks(count: int) -> list[HarborTask]:
    # load_harbor_tasks_from_dir returns tasks sorted by task_name, so mirror that
    # ordering here: the seeded shuffle is only reproducible given a stable input.
    return [
        HarborTask(task_name=f"task-{index:03d}", instruction="do it", task_dir=None)  # type: ignore[arg-type]
        for index in range(count)
    ]


def _builder(count: int = 100, **overrides) -> HarborDatasetBuilder:
    kwargs = {
        "tasks": _tasks(count),
        "batch_size": 2,
        "group_size": 4,
        "model_name": MODEL,
    }
    kwargs.update(overrides)
    return HarborDatasetBuilder(**kwargs)  # type: ignore[arg-type]


class TestHeldOutSplit:
    def test_split_is_disjoint_and_covers_every_task(self) -> None:
        train, held_out = _builder(eval_size=20, seed=7).split_tasks()

        assert len(train) == 80
        assert len(held_out) == 20
        train_names = {task.task_name for task in train}
        held_names = {task.task_name for task in held_out}
        assert not (train_names & held_names)
        assert len(train_names | held_names) == 100

    def test_split_is_deterministic_for_a_fixed_seed(self) -> None:
        first = _builder(eval_size=20, seed=7).split_tasks()[1]
        second = _builder(eval_size=20, seed=7).split_tasks()[1]
        other = _builder(eval_size=20, seed=8).split_tasks()[1]

        assert [t.task_name for t in first] == [t.task_name for t in second]
        assert [t.task_name for t in first] != [t.task_name for t in other]

    def test_eval_size_zero_keeps_legacy_in_sample_eval(self) -> None:
        builder = _builder()
        train, held_out = builder.split_tasks()

        assert len(train) == 100
        assert held_out == []
        # The legacy contract: a non-None test dataset over every task.
        _, eval_dataset = asyncio.run(builder())
        assert eval_dataset is not None
        assert builder.make_held_out_env_group_builders() == []

    def test_held_out_split_returns_no_auto_wrapped_eval_dataset(self) -> None:
        """rl.train auto-wraps the second tuple element; a held-out run must opt out
        so it does not also spin up an in-sample eval over every task."""
        _, eval_dataset = asyncio.run(_builder(eval_size=20, seed=7)())

        assert eval_dataset is None

    def test_held_out_builders_use_eval_group_size(self) -> None:
        builders = _builder(
            eval_size=20, eval_group_size=4, seed=7
        ).make_held_out_env_group_builders()

        assert len(builders) == 20
        assert {builder.group_size for builder in builders} == {4}

    def test_train_dataset_excludes_held_out_tasks(self) -> None:
        builder = _builder(eval_size=20, seed=7)
        _, held_out = builder.split_tasks()
        held_names = {task.task_name for task in held_out}

        train_dataset, _ = asyncio.run(builder())
        batched = [
            group_builder
            for index in range(len(train_dataset))
            for group_builder in train_dataset.get_batch(index)
        ]

        assert len(batched) == 80
        assert not {b.task.task_name for b in batched} & held_names

    @pytest.mark.parametrize("eval_size", [100, 101, -1])
    def test_out_of_range_eval_size_is_rejected(self, eval_size: int) -> None:
        with pytest.raises(ValueError, match="eval_size must be in"):
            _builder(eval_size=eval_size).split_tasks()

    def test_raise_on_grading_error_reaches_env_group_builders(self) -> None:
        builders = _builder(
            eval_size=4, seed=1, raise_on_grading_error=True
        ).make_held_out_env_group_builders()

        assert all(builder.raise_on_grading_error for builder in builders)


class TestBoundedConcurrencyEvaluator:
    def test_is_an_rl_test_set_evaluator(self) -> None:
        """rl.train special-cases RLTestSetEvaluator for rollout export and log naming,
        so the held-out evaluator must remain an instance of it."""
        evaluator = BoundedConcurrencyEvaluator([], max_tokens=1)

        assert isinstance(evaluator, RLTestSetEvaluator)
        assert evaluator.name == "heldout"

    def test_every_held_out_group_is_rolled_out(self) -> None:
        builders = _builder(
            eval_size=20, eval_group_size=4, seed=7
        ).make_held_out_env_group_builders()
        evaluator = BoundedConcurrencyEvaluator(builders, max_tokens=1, max_concurrent_groups=2)

        assert len(evaluator.env_group_builders_P) == 20

    def test_concurrency_never_exceeds_the_limit(self) -> None:
        """The whole point of the class: peak in-flight groups stays at the limit,
        which is what bounds sandbox count to limit * group_size."""
        builders = _builder(
            eval_size=20, eval_group_size=4, seed=7
        ).make_held_out_env_group_builders()
        evaluator = BoundedConcurrencyEvaluator(builders, max_tokens=1, max_concurrent_groups=3)

        in_flight = 0
        peak = 0
        completed = 0

        async def fake_run_one_group(builder, group_idx, policy):
            nonlocal in_flight, peak, completed
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)  # yield so overlapping rollouts can pile up
            in_flight -= 1
            completed += 1
            return None

        evaluator._run_one_group = fake_run_one_group  # type: ignore[method-assign]
        # All groups fail, so _collect_eval_metrics reports errors and no rewards.
        metrics = asyncio.run(evaluator.eval_token_completer(policy=None))  # type: ignore[arg-type]

        assert completed == 20
        assert peak <= 3, f"peak concurrency {peak} exceeded the limit of 3"
        assert metrics  # error metrics are still emitted

    def test_metrics_are_computed_once_over_all_groups(self) -> None:
        """Aggregating per-chunk metric dicts would corrupt sums and per-turn means,
        so there must be exactly one _collect_eval_metrics call."""
        builders = _builder(eval_size=6, seed=7).make_held_out_env_group_builders()
        evaluator = BoundedConcurrencyEvaluator(builders, max_tokens=1, max_concurrent_groups=2)

        calls: list[int] = []

        async def fake_run_one_group(builder, group_idx, policy):
            return None

        def fake_collect(results, rollout_summary_export, *, store=None):
            calls.append(len(results))
            return {"heldout/env/all/reward/total": 0.0}

        evaluator._run_one_group = fake_run_one_group  # type: ignore[method-assign]
        evaluator._collect_eval_metrics = fake_collect  # type: ignore[method-assign]
        asyncio.run(evaluator.eval_token_completer(policy=None))  # type: ignore[arg-type]

        assert calls == [6]

    def test_max_concurrent_groups_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_groups must be at least 1"):
            BoundedConcurrencyEvaluator([], max_tokens=1, max_concurrent_groups=0)


class TestRolloutBudget:
    """The runner enforces ``EnvFromMessageEnv.rollout_limits``, so the budgets
    configured here must land there -- not be replaced by the model preset's
    defaults (Inkling: 10 turns / 30 tool calls / 64K tokens)."""

    def test_configured_budgets_reach_the_runner(self) -> None:
        from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborEnvGroupBuilder
        from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

        class FakeSandbox:
            async def run_command(self, *args, **kwargs):
                return SandboxResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, *args, **kwargs):
                return SandboxResult(exit_code=0, stdout="", stderr="")

            async def write_file(self, *args, **kwargs):
                return SandboxResult(exit_code=0, stdout="", stderr="")

            async def send_heartbeat(self) -> None: ...

            async def cleanup(self) -> None: ...

        async def fake_factory(env_dir, timeout):
            return FakeSandbox()

        task = HarborTask(
            task_name="t",
            instruction="do it",
            task_dir=Path("data/kokkos/SWE-kokkos-bench-v2/kokkos__kokkos-6289"),
        )
        builder = HarborEnvGroupBuilder(
            task=task,
            model_name=MODEL,
            renderer_name=None,
            max_turns=40,
            group_size=1,
            max_trajectory_tokens=112 * 1024,
            max_tool_calls=80,
            sandbox_factory=fake_factory,
        )

        env = asyncio.run(builder.make_envs())[0]

        assert env.rollout_limits is not None
        assert env.rollout_limits.max_turns == 40
        assert env.rollout_limits.max_tool_calls == 80
        assert env.rollout_limits.max_trajectory_tokens == 112 * 1024

    def test_hitting_the_turn_cap_does_not_zero_a_passing_reward(self) -> None:
        """A Harbor task is graded by running test.sh against the repo state, so a
        trajectory that fixed the bug has earned its reward even if it spent its
        last turn on one more read-only command. Clamping those to zero measured
        22 of 71 verified successes away in a 20-step run."""
        from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborEnvGroupBuilder
        from tinker_cookbook.rl.rollout_presets import agentic
        from tinker_cookbook.rl.types import StopReason

        builder = HarborEnvGroupBuilder(
            task=None,  # type: ignore[arg-type]
            model_name=MODEL,
            renderer_name=None,
            max_turns=40,
            group_size=1,
            max_tool_calls=80,
        )
        reasons = builder._rollout_config().termination.limit_stop_reasons

        assert StopReason.MAX_TURNS not in reasons
        # The other budget limits still clamp: a token-exhausted trajectory
        # really is unfinished.
        for kept in (
            StopReason.MAX_TOKENS,
            StopReason.MAX_SAMPLED_TOKENS,
            StopReason.MAX_TOOL_CALLS,
            StopReason.ROLLOUT_TIMEOUT,
        ):
            assert kept in reasons
        # The shared preset must not be mutated for other recipes.
        assert StopReason.MAX_TURNS in agentic().termination.limit_stop_reasons

    def test_trainer_side_termination_is_also_unclamped(self) -> None:
        """The clamp is applied in `do_group_rollout` from
        `Config.effective_termination()`, NOT from the env's rollout_config. Fixing
        only the env side leaves it active, because the trainer falls back to the
        model preset — which is exactly what happened on the first fix attempt."""
        from tinker_cookbook.recipes.harbor_rl.harbor_env import harbor_termination_policy
        from tinker_cookbook.rl.train import Config
        from tinker_cookbook.rl.types import StopReason

        builder = _builder(count=3)

        unset = Config(
            learning_rate=1e-5,
            dataset_builder=builder,
            model_name=MODEL,
            recipe_name="r",
            max_tokens=1,
            log_path="/tmp/x",
        )
        # Without an explicit policy the Inkling preset clamps MAX_TURNS.
        assert StopReason.MAX_TURNS in unset.effective_termination().limit_stop_reasons

        wired = Config(
            learning_rate=1e-5,
            dataset_builder=builder,
            model_name=MODEL,
            recipe_name="r",
            max_tokens=1,
            log_path="/tmp/x",
            termination=harbor_termination_policy(),
        )
        effective = wired.effective_termination()
        assert StopReason.MAX_TURNS not in effective.limit_stop_reasons
        assert effective.zero_reward_on_limit is True


class TestBoundedHeldOutEvaluatorBuilder:
    def test_builder_forwards_its_configuration(self) -> None:
        builders = _builder(
            eval_size=8, eval_group_size=4, seed=3
        ).make_held_out_env_group_builders()
        evaluator = BoundedHeldOutEvaluatorBuilder(
            env_group_builders=builders,
            max_tokens=16384,
            max_concurrent_groups=2,
        )()

        assert isinstance(evaluator, BoundedConcurrencyEvaluator)
        assert len(evaluator.env_group_builders_P) == 8
        assert evaluator.max_tokens == 16384
        assert evaluator.max_concurrent_groups == 2
        assert evaluator.name == "heldout"

    def test_a_fault_tolerant_strategy_survives_group_errors(self, monkeypatch) -> None:
        """Without an error-catching strategy, one grading failure during eval
        propagates out of asyncio.gather and kills the whole training run."""
        from tinker_cookbook.rl.rollout_presets import default_rollout_strategy_for_model

        builders = _builder(eval_size=4, seed=3).make_held_out_env_group_builders()
        evaluator = BoundedHeldOutEvaluatorBuilder(
            env_group_builders=builders,
            max_tokens=1,
            strategy=default_rollout_strategy_for_model(MODEL),
        )()

        assert evaluator.strategy is not None
        assert evaluator.strategy.catches_group_errors

        # Fail inside the rollout itself, below _run_one_group's try/except, so the
        # real error handling is what gets exercised.
        async def boom(*args, **kwargs):
            raise RuntimeError("verifier command failed before producing a result")

        monkeypatch.setattr("tinker_cookbook.rl.rollouts.do_group_rollout", boom)

        collected: list[list[object]] = []

        def fake_collect(results, rollout_summary_export, *, store=None):
            collected.append(results)
            return {}

        evaluator._collect_eval_metrics = fake_collect  # type: ignore[method-assign]
        asyncio.run(evaluator.eval_token_completer(policy=None))  # type: ignore[arg-type]

        assert collected and collected[0] == [None, None, None, None]

    def test_without_a_strategy_group_errors_propagate(self, monkeypatch) -> None:
        """The failure mode this fix addresses: strategy=None re-raises, so a single
        ConTree grading failure takes down the run."""
        builders = _builder(eval_size=2, seed=3).make_held_out_env_group_builders()
        evaluator = BoundedHeldOutEvaluatorBuilder(env_group_builders=builders, max_tokens=1)()

        async def boom(*args, **kwargs):
            raise RuntimeError("Image state cannot have state FAILED")

        monkeypatch.setattr("tinker_cookbook.rl.rollouts.do_group_rollout", boom)

        assert evaluator.strategy is None
        with pytest.raises(RuntimeError, match="cannot have state FAILED"):
            asyncio.run(evaluator.eval_token_completer(policy=None))  # type: ignore[arg-type]

    def test_config_dump_summarises_instead_of_inlining_tasks(self) -> None:
        """The run config is serialized into experiment trackers. Inlining 20 tasks'
        instruction text bloats it by tens of KB and duplicates dataset_builder;
        wandb rejected the oversized dump outright."""
        from tinker_cookbook.utils.ml_log import dump_config

        builders = _builder(
            eval_size=20, eval_group_size=4, seed=7
        ).make_held_out_env_group_builders()
        builder = BoundedHeldOutEvaluatorBuilder(
            env_group_builders=builders, max_tokens=16384, max_concurrent_groups=2
        )

        dumped = dump_config(builder)

        assert dumped["num_held_out_tasks"] == 20
        assert dumped["max_concurrent_groups"] == 2
        assert len(dumped["task_names"]) == 20
        # The tasks themselves must not be inlined.
        assert "env_group_builders" not in dumped
        assert "instruction" not in json.dumps(dumped)
