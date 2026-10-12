"""A dollar cap shared by every process of one campaign.

Each model request reserves its worst-case cost before it is sent and settles to the
priced usage afterwards, under a file lock, so parallel shards cannot together exceed the
cap. Prices are per million tokens; the defaults are Tinker's listed Inkling-Small (256K)
prices (https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/,
read 2026-10-12, limited-time discount). Re-check them before a campaign.
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class BudgetStop(Exception):
    """The campaign's dollar cap would be exceeded."""


@dataclass(frozen=True)
class Prices:
    prefill: float = 0.58
    cached_prefill: float = 0.116
    sample: float = 1.44
    train: float = 3.47

    def request_cost(self, prompt_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        uncached = max(0, prompt_tokens - cached_tokens)
        return (
            uncached * self.prefill
            + cached_tokens * self.cached_prefill
            + output_tokens * self.sample
        ) / 1e6


class SpendLedger:
    def __init__(self, path: Path, cap_usd: float, prices: Prices = Prices()) -> None:
        self.path = path
        self.cap_usd = cap_usd
        self.prices = prices
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as state:
            if state.get("cap_usd") not in (None, cap_usd):
                raise ValueError(f"ledger cap is {state['cap_usd']}, not {cap_usd}")
            state["cap_usd"] = cap_usd

    @contextmanager
    def _locked(self):
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            state.setdefault("settled_usd", 0.0)
            state.setdefault("reserved", {})
            state.setdefault("entries", 0)
            yield state
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(self.path)

    def reserve(self, estimate_usd: float, label: str) -> str:
        with self._locked() as state:
            committed = state["settled_usd"] + sum(r["usd"] for r in state["reserved"].values())
            if committed + estimate_usd > self.cap_usd:
                raise BudgetStop(
                    f"{label}: ${committed:.2f} committed + ${estimate_usd:.2f} would exceed "
                    f"the ${self.cap_usd:.2f} cap"
                )
            token = uuid.uuid4().hex
            state["reserved"][token] = {"usd": estimate_usd, "label": label, "time": time.time()}
            return token

    def settle(self, token: str, actual_usd: float) -> None:
        with self._locked() as state:
            state["reserved"].pop(token, None)
            state["settled_usd"] += actual_usd
            state["entries"] += 1

    def spent(self) -> float:
        with self._locked() as state:
            return state["settled_usd"]


def capped_chat_client(client, ledger: SpendLedger, *, label: str, max_prompt_tokens: int):
    """Wrap ``client.chat.completions.create`` so every request goes through the ledger.

    The reservation assumes an uncached prompt of ``max_prompt_tokens`` and the request's
    full ``max_tokens`` of output; a failed request settles at that reservation, since the
    provider may still bill it.
    """
    create = client.chat.completions.create
    prices = ledger.prices

    async def guarded(**kwargs):
        estimate = prices.request_cost(max_prompt_tokens, 0, int(kwargs.get("max_tokens") or 0))
        token = ledger.reserve(estimate, label)
        try:
            response = await create(**kwargs)
        except BaseException:
            ledger.settle(token, estimate)
            raise
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        ledger.settle(
            token, prices.request_cost(usage.prompt_tokens, cached, usage.completion_tokens)
        )
        return response

    client.chat.completions.create = guarded
    return client
