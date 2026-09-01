from types import SimpleNamespace

from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import _function_calls, _usage


def test_function_calls_filters_non_tool_output() -> None:
    call = SimpleNamespace(type="function_call")
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning"), call])
    assert _function_calls(response) == [call]


def test_usage_extracts_reasoning_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )
    assert _usage(response) == (12, 7, 3)
