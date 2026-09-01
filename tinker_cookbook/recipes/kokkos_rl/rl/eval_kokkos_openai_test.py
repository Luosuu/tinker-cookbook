from types import SimpleNamespace

from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import (
    CLIConfig,
    _function_calls,
    _request_cost_usd,
    _truncate_tool_output,
    _usage,
)


def test_function_calls_filters_non_tool_output() -> None:
    call = SimpleNamespace(type="function_call")
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning"), call])
    assert _function_calls(response) == [call]


def test_usage_extracts_reasoning_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=5, cache_write_tokens=4),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )
    assert _usage(response) == (12, 7, 3, 5, 4)


def test_request_cost_distinguishes_cache_reads_and_writes() -> None:
    config = CLIConfig(
        input_price_per_million=2.0,
        cached_input_price_per_million=0.2,
        cache_write_price_per_million=2.5,
        output_price_per_million=12.0,
    )
    cost = _request_cost_usd(
        input_tokens=1_000_000,
        cached_input_tokens=600_000,
        cache_write_input_tokens=100_000,
        output_tokens=100_000,
        config=config,
    )
    assert cost == 2.17


def test_tool_output_truncation_preserves_head_and_tail() -> None:
    output = "a" * 100 + " important tail"
    truncated = _truncate_tool_output(output, 60)
    assert len(truncated) == 60
    assert truncated.startswith("a")
    assert truncated.endswith(" important tail")
    assert "characters omitted" in truncated
