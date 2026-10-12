import asyncio
from types import SimpleNamespace

import pytest

from tinker_cookbook.recipes.kokkos_rl.spend import (
    BudgetStop,
    Prices,
    SpendLedger,
    capped_chat_client,
)


def test_prices_split_cached_and_uncached_prompt():
    prices = Prices(prefill=0.58, cached_prefill=0.116, sample=1.44)
    assert prices.request_cost(1_000_000, 900_000, 10_000) == pytest.approx(
        0.1 * 0.58 + 0.9 * 0.116 + 0.01 * 1.44
    )


def test_reservations_bound_parallel_spend(tmp_path):
    first = SpendLedger(tmp_path / "ledger.json", cap_usd=1.0)
    second = SpendLedger(tmp_path / "ledger.json", cap_usd=1.0)
    a = first.reserve(0.6, "a")
    with pytest.raises(BudgetStop, match="would exceed"):
        second.reserve(0.5, "b")
    first.settle(a, 0.2)
    second.settle(second.reserve(0.5, "b"), 0.5)
    assert first.spent() == pytest.approx(0.7)
    with pytest.raises(ValueError, match="cap"):
        SpendLedger(tmp_path / "ledger.json", cap_usd=2.0)


def test_chat_client_settles_actual_usage_and_failures_at_the_reservation(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.json", cap_usd=10.0, prices=Prices(1.0, 0.1, 2.0))
    usage = SimpleNamespace(
        prompt_tokens=100_000,
        completion_tokens=1_000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=50_000),
    )
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fail"):
            raise RuntimeError("502")
        return SimpleNamespace(usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    capped_chat_client(client, ledger, label="t", max_prompt_tokens=200_000)
    asyncio.run(client.chat.completions.create(max_tokens=4096))
    assert ledger.spent() == pytest.approx((50_000 * 1.0 + 50_000 * 0.1 + 1_000 * 2.0) / 1e6)
    with pytest.raises(RuntimeError):
        asyncio.run(client.chat.completions.create(max_tokens=4096, fail=True))
    assert ledger.spent() == pytest.approx(0.057 + (200_000 * 1.0 + 4096 * 2.0) / 1e6)
