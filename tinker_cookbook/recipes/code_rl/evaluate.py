from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.code_rl.code_grading import close_cpp_sandbox_pools
from tinker_cookbook.recipes.code_rl.livecodebench_cpp import (
    DATASET_REVISION,
    DatasetSplit,
    LiveCodeBenchCppDatasetBuilder,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.git_rev import recipe_user_metadata


@chz.chz
class CLIConfig:
    """Evaluate a base model or checkpoint on a held-out LiveCodeBench-CPP split."""

    model_name: str = "openai/gpt-oss-20b"
    checkpoint_path: str | None = None
    renderer_name: str | None = None
    base_url: str | None = None

    dataset_split: DatasetSplit = "v6_2408_2505"
    dataset_revision: str = DATASET_REVISION
    seed: int = 0
    eval_size: int = 32
    max_eval_examples: int | None = None
    batch_size: int = 32

    max_tokens: int = 32768
    max_turns: int = 2
    cpp_timeout: int = 30
    sandbox_backend: SandboxBackend = SandboxBackend.SANDBOXFUSION
    rollout_max_retries: int = 1

    output_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(config: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=config.model_name,
        explicit_renderer_name=config.renderer_name,
        load_checkpoint_path=config.checkpoint_path,
        base_url=config.base_url,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_path = config.output_path or (
        f"/tmp/tinker-examples/code_rl/eval-livecodebench-cpp-{timestamp}"
    )
    cli_utils.check_log_dir(
        output_path,
        behavior_if_exists=config.behavior_if_log_dir_exists,
    )
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = LiveCodeBenchCppDatasetBuilder(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        batch_size=config.batch_size,
        group_size=1,
        dataset_split=config.dataset_split,
        dataset_revision=config.dataset_revision,
        eval_size=config.eval_size,
        max_train_examples=1,
        max_eval_examples=config.max_eval_examples,
        max_turns=config.max_turns,
        max_generation_tokens=config.max_tokens,
        timeout=config.cpp_timeout,
        sandbox_backend=config.sandbox_backend,
        seed=config.seed,
    )
    _, eval_dataset = await builder()
    if eval_dataset is None:
        raise ValueError("Evaluation set is empty")

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("eval_livecodebench_cpp"),
    )
    if config.checkpoint_path:
        sampling_client = service_client.create_sampling_client(
            model_path=config.checkpoint_path,
            base_model=config.model_name,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    evaluator = RLTestSetEvaluator(
        eval_dataset,
        max_tokens=config.max_tokens,
        name="livecodebench_cpp",
        strategy=RetryOnFailure(max_retries=config.rollout_max_retries),
    )
    report_path = output_dir / "rollouts.html"
    try:
        with logtree.init_trace("LiveCodeBench-CPP evaluation", path=report_path):
            metrics = await evaluator(sampling_client)
    finally:
        await close_cpp_sandbox_pools()

    result = evaluator.last_result
    if result is None:
        raise RuntimeError("Evaluator did not produce a result")
    payload = {
        "config": {
            "model_name": config.model_name,
            "checkpoint_path": config.checkpoint_path,
            "renderer_name": renderer_name,
            "dataset_split": config.dataset_split,
            "dataset_revision": config.dataset_revision,
            "seed": config.seed,
            "eval_size": config.eval_size,
            "max_eval_examples": config.max_eval_examples,
            "max_tokens": config.max_tokens,
            "sandbox_backend": config.sandbox_backend.value,
        },
        "result": asdict(result),
        "metrics": metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
