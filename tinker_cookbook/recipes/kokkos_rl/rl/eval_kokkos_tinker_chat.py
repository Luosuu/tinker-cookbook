"""Forty-turn Kokkos evaluation through Tinker's hosted Chat Completions API."""

import asyncio

import chz

from tinker_cookbook.recipes.kokkos_rl.chat_inference import TINKER_CHAT_BASE_URL
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import CLIConfig, main


@chz.chz
class TinkerChatConfig(CLIConfig):
    api_mode: str = "chat"
    base_url: str | None = TINKER_CHAT_BASE_URL
    api_key_env: str = "TINKER_API_KEY"
    model_name: str = "thinkingmachines/Inkling-Small:peft:262144"
    output_path: str = "notes/experiments/SWE-kokkos-bench/tinker-chat/pass-at-1"
    max_turns: int = 40
    max_tokens: int = 16384
    max_sampled_tokens: int = 65536
    max_input_tokens: int = 5_000_000
    max_tool_calls: int = 80
    grader_timeout: int = 900
    sandbox_build_parallelism: int | None = 1
    # Preserve token/turn budgets without misapplying another provider's prices.
    estimate_cost: bool = False
    max_cost_usd_per_task: float | None = None
    max_infra_retries: int = 0


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(TinkerChatConfig)))
