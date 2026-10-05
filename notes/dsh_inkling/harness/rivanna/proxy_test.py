"""Tests for the comparability proxy.

The proxy exists to guarantee two properties that a config file cannot: every upstream
request carries the reference inference parameters, and an episode stops at the reference
model-request cap. Both are load-bearing for comparing a DSH score against the native
baseline, so both are pinned here.
"""

from __future__ import annotations

import json

import pytest

from notes.dsh_inkling.harness.rivanna.proxy import (
    REFERENCE_REASONING_EFFORT,
    REFERENCE_SEPARATE_REASONING,
    Budget,
    _inject_reference_inference,
    _join_upstream,
)


def test_injects_reference_inference_parameters() -> None:
    body = json.dumps({"model": "m", "messages": []}).encode()
    payload = json.loads(_inject_reference_inference(body))
    assert payload["reasoning_effort"] == REFERENCE_REASONING_EFFORT
    assert payload["separate_reasoning"] is REFERENCE_SEPARATE_REASONING
    assert payload["temperature"] == 1


def test_reference_inference_overrides_harness_values() -> None:
    """A harness that sends its own values must not be able to drift the comparison."""
    body = json.dumps(
        {"model": "m", "reasoning_effort": "high", "temperature": 0.2}
    ).encode()
    payload = json.loads(_inject_reference_inference(body))
    assert payload["reasoning_effort"] == REFERENCE_REASONING_EFFORT
    assert payload["temperature"] == 1


def test_injection_preserves_other_fields() -> None:
    body = json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 7}
    ).encode()
    payload = json.loads(_inject_reference_inference(body))
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 7
    assert payload["messages"] == [{"role": "user", "content": "x"}]


def test_injection_handles_empty_body() -> None:
    assert json.loads(_inject_reference_inference(b""))["reasoning_effort"] == (
        REFERENCE_REASONING_EFFORT
    )


def test_budget_admits_exactly_max_requests() -> None:
    budget = Budget(max_requests=3, max_input_tokens=10**9)
    assert [budget.admit()[0] for _ in range(4)] == [True, True, True, False]


def test_budget_reports_reason_when_exhausted() -> None:
    budget = Budget(max_requests=0, max_input_tokens=10**9)
    allowed, reason = budget.admit()
    assert not allowed
    assert "budget exhausted" in reason


def test_budget_stops_on_input_tokens() -> None:
    budget = Budget(max_requests=100, max_input_tokens=100)
    budget.record_usage({"prompt_tokens": 120, "completion_tokens": 5})
    allowed, reason = budget.admit()
    assert not allowed
    assert "input-token budget" in reason


def test_budget_records_usage() -> None:
    budget = Budget(max_requests=10, max_input_tokens=10**9)
    budget.record_usage({"prompt_tokens": 11, "completion_tokens": 3})
    budget.record_usage({"prompt_tokens": 4, "completion_tokens": 2})
    snapshot = budget.snapshot()
    assert snapshot["input_tokens"] == 15
    assert snapshot["output_tokens"] == 5


@pytest.mark.parametrize(
    "base,path,expected",
    [
        ("https://host/v1", "/v1/chat/completions", "https://host/v1/chat/completions"),
        ("https://host/v1/", "/v1/chat/completions", "https://host/v1/chat/completions"),
        ("https://host", "/v1/chat/completions", "https://host/v1/chat/completions"),
        ("https://host/v1", "/v1/models", "https://host/v1/models"),
        ("https://host/api/v1", "/v1/chat/completions", "https://host/api/v1/chat/completions"),
    ],
)
def test_upstream_join_never_doubles_version(base: str, path: str, expected: str) -> None:
    assert _join_upstream(base, path) == expected
