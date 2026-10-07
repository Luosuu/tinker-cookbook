"""Run the native Chat Completions arm on ConTree.

The audited reference score of 52/100 was produced by the native tool loop on Rivanna,
while the DSH arm runs on ConTree. Comparing them confounds the harness with the sandbox
and rests on a single draw on each side. This runs the native loop on ConTree with the same
task payloads and budget, so the harness is the only difference and the comparison is
paired by task.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    main,
)


def _config(tasks_dir: str, output_path: str, cache_path: str, max_concurrency: int) -> CLIConfig:
    # chz dataclasses are not stdlib dataclasses, so construct directly rather than
    # using dataclasses.replace. Match the audited baseline budget exactly: 40 model
    # turns, 80 tool calls, 65,536 sampled tokens, effort 0.9, temperature 1, no network.
    return CLIConfig(
        model_name="thinkingmachines/Inkling-Small:peft:262144",
        tasks_dir=tasks_dir,
        output_path=output_path,
        env_file=".env",
        api_mode="chat",
        chat_provider="tinker",
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key_env="TINKER_API_KEY",
        thinking_effort=0.9,
        reasoning_effort="high",
        temperature=1.0,
        max_turns=40,
        max_tokens=16384,
        max_sampled_tokens=65536,
        max_input_tokens=5_000_000,
        max_tool_calls=80,
        max_tool_output_chars=16384,
        max_cost_usd_per_task=None,
        estimate_cost=False,
        # ConTree rejects anything above 3600 (instance_max_timeout).
        sandbox_timeout=3600,
        command_timeout=900,
        grader_timeout=1200,
        max_concurrency=max_concurrency,
        max_infra_retries=0,
        allow_network=False,
        contree_cache_path=cache_path,
        # Two CPUs halves the verifier's build without OOM on a 4 GB sandbox.
        sandbox_build_parallelism=2,
    )


if __name__ == "__main__":
    tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "data/kokkos/SWE-kokkos-bench-v2"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dsh-contree/native01"
    cache = sys.argv[3] if len(sys.argv) > 3 else "/tmp/dsh-contree/cache/images.json"
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    load_env_file(Path(".env"))
    asyncio.run(main(_config(tasks_dir, output_path, cache, concurrency)))
