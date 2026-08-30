"""Held-out evaluation for Harbor RL with bounded sandbox concurrency.

:class:`~tinker_cookbook.rl.metric_util.RLTestSetEvaluator` rolls out every group in
a single ``asyncio.gather``, so its peak concurrency is the size of the whole eval set
-- a dataset's ``batch_size`` does not bound it, because the evaluator flattens the
dataset up front. Harbor tasks each hold a container sandbox and cloud backends degrade
sharply under load, so a held-out set of any useful size needs an explicit limit.

:class:`BoundedConcurrencyEvaluator` adds that limit and changes nothing else: metrics
are still computed once over every group, so sums stay sums and per-turn means stay
weighted by turns.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import tinker

from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, RolloutSummaryExportConfig
from tinker_cookbook.rl.rollout_strategy import RolloutStrategy
from tinker_cookbook.rl.types import EnvGroupBuilder

logger = logging.getLogger(__name__)


class BoundedConcurrencyEvaluator(RLTestSetEvaluator):
    """An ``RLTestSetEvaluator`` that rolls out at most N groups at a time.

    Takes env group builders directly rather than an ``RLDataset``: the base class
    only accepts a dataset in order to flatten it, and batching is irrelevant here.

    Peak sandbox concurrency is ``max_concurrent_groups * group_size`` where
    ``group_size`` is whatever the builders were constructed with.

    Args:
        env_group_builders: One builder per held-out task.
        max_tokens: Maximum tokens per completion.
        name: Metric namespace, e.g. ``"heldout"`` yields ``heldout/...`` keys.
        max_concurrent_groups: Groups rolled out concurrently.
        num_groups_to_log: Leading groups for which full logtree logging is enabled.
        strategy: Optional rollout error-handling strategy.
    """

    def __init__(
        self,
        env_group_builders: Sequence[EnvGroupBuilder],
        max_tokens: int,
        name: str = "heldout",
        max_concurrent_groups: int = 2,
        num_groups_to_log: int = 4,
        strategy: RolloutStrategy | None = None,
    ):
        if max_concurrent_groups < 1:
            raise ValueError(
                f"max_concurrent_groups must be at least 1, got {max_concurrent_groups}"
            )
        # Deliberately not calling super().__init__: it exists to flatten an
        # RLDataset into exactly the list we already hold.
        self.env_group_builders_P = list(env_group_builders)
        self.max_tokens = max_tokens
        self.name = name
        self.max_concurrent_groups = max_concurrent_groups
        self.num_groups_to_log = num_groups_to_log
        self.strategy = strategy
        self.last_result = None

    async def eval_token_completer(
        self,
        policy,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
        store=None,
    ) -> dict[str, float]:
        """Roll out every group under a concurrency limit, then score them together."""
        semaphore = asyncio.Semaphore(self.max_concurrent_groups)
        logger.info(
            "Held-out eval: %d groups, at most %d concurrent",
            len(self.env_group_builders_P),
            self.max_concurrent_groups,
        )

        async def run_bounded(builder: EnvGroupBuilder, group_idx: int):
            async with semaphore:
                return await self._run_one_group(builder, group_idx, policy)

        results = await asyncio.gather(
            *[
                run_bounded(builder, group_idx)
                for group_idx, builder in enumerate(self.env_group_builders_P)
            ]
        )
        return self._collect_eval_metrics(results, rollout_summary_export, store=store)

    async def _run_one_group(self, builder: EnvGroupBuilder, group_idx: int, policy):
        """Roll out a single group, applying the same error handling as the base class."""
        from tinker_cookbook.exceptions import AllTrajectoriesFailedError
        from tinker_cookbook.rl.rollouts import do_group_rollout
        from tinker_cookbook.utils import logtree

        try:
            with logtree.optional_enable_logging(enable=group_idx < self.num_groups_to_log):
                return await do_group_rollout(builder, policy, strategy=self.strategy)
        except AllTrajectoriesFailedError as e:
            logger.warning(f"Held-out eval: {e}")
            return None
        except Exception as e:
            if self.strategy is None or not self.strategy.catches_group_errors:
                raise
            logger.warning(f"Held-out eval rollout error ({type(e).__name__}): {e}")
            return None

    async def _eval_with_executor(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        rollout_summary_export: RolloutSummaryExportConfig | None = None,
        store=None,
    ) -> dict[str, float]:
        """Bound concurrency on the executor path too.

        Without this override, registering a rollout executor would silently restore
        the unbounded fan-out this class exists to prevent.
        """
        from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward

        semaphore = asyncio.Semaphore(self.max_concurrent_groups)

        async def run_bounded(builder: EnvGroupBuilder, group_idx: int):
            async with semaphore:
                return await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    builder,
                    max_tokens=self.max_tokens,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    enable_logging=group_idx < self.num_groups_to_log,
                    strategy=self.strategy,
                )

        results = await asyncio.gather(
            *[
                run_bounded(builder, group_idx)
                for group_idx, builder in enumerate(self.env_group_builders_P)
            ]
        )
        return self._collect_eval_metrics(results, rollout_summary_export, store=store)


class BoundedHeldOutEvaluatorBuilder:
    """Builder that constructs a :class:`BoundedConcurrencyEvaluator`.

    Holds configuration only; the training loop calls it once at startup.

    Deliberately a plain class rather than a ``@chz.chz`` dataclass: the training
    config is serialized into experiment trackers, and each held-out
    ``HarborEnvGroupBuilder`` carries a whole task (instruction text and TOML
    config). Serializing those would bloat the run config and duplicate the task
    payloads already recorded under ``dataset_builder``.
    """

    def __init__(
        self,
        env_group_builders: Sequence[EnvGroupBuilder],
        max_tokens: int,
        name: str = "heldout",
        max_concurrent_groups: int = 2,
        strategy: RolloutStrategy | None = None,
    ):
        self.env_group_builders = list(env_group_builders)
        self.max_tokens = max_tokens
        self.name = name
        self.max_concurrent_groups = max_concurrent_groups
        self.strategy = strategy

    def to_dict(self) -> dict[str, object]:
        """Summarise for config dumps instead of inlining the held-out tasks.

        ``ml_log.dump_config`` prefers ``to_dict``; without it the dump recurses
        into every ``HarborEnvGroupBuilder`` and inlines each task's instruction
        text and TOML config -- hundreds of KB, duplicating what is already
        recorded under ``dataset_builder``, and large enough that trackers reject
        the config.
        """
        return {
            "name": self.name,
            "num_held_out_tasks": len(self.env_group_builders),
            "max_tokens": self.max_tokens,
            "max_concurrent_groups": self.max_concurrent_groups,
            "strategy": type(self.strategy).__name__ if self.strategy else None,
            "task_names": [
                getattr(getattr(b, "task", None), "task_name", "?")
                for b in self.env_group_builders
            ],
        }

    def __repr__(self) -> str:
        # Keep config dumps small and stable: summarise, never inline the tasks.
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"num_tasks={len(self.env_group_builders)}, "
            f"max_concurrent_groups={self.max_concurrent_groups})"
        )

    def __call__(self) -> BoundedConcurrencyEvaluator:
        return BoundedConcurrencyEvaluator(
            env_group_builders=self.env_group_builders,
            max_tokens=self.max_tokens,
            name=self.name,
            max_concurrent_groups=self.max_concurrent_groups,
            strategy=self.strategy,
        )
