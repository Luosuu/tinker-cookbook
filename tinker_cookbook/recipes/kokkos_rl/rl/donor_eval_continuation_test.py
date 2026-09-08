"""No-repeat scope and raw parse-error provenance for an independent continuation."""

import asyncio
import json
from dataclasses import asdict

import pytest

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval_continuation as continuation


def scope_fixture(tmp_path):
    all_slots = [
        {"arm": arm, "task_name": f"task{n:02d}", "sample_index": i, "task_sha256": "hash"}
        for arm in ("random_success", "short_success")
        for i in range(4)
        for n in range(20)
    ]
    claims = all_slots[:22]
    audit = {"slots": []}
    rows = []
    for s in claims:
        folder = tmp_path / s["arm"] / "attempts"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{s['task_name']}__{s['sample_index']:02d}.json").write_text(json.dumps(s))
        r = TaskResult(s["task_name"], s["sample_index"], 0, {}, 1, 1)
        rows.append(asdict(r))
        audit["slots"].append({"arm": s["arm"], "result": asdict(r)})
    (tmp_path / "random_success/results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    excluded = sorted(
        [
            {"arm": s["arm"], "task_name": s["task_name"], "sample_index": s["sample_index"]}
            for s in claims
        ],
        key=lambda s: (s["arm"], s["task_name"], s["sample_index"]),
    )
    proposal = {
        "slots": all_slots[22:],
        "excluded_prior_claims": excluded,
        "maximum_new_rollouts": 138,
        "counts": {"random_success": 58, "short_success": 80},
    }
    return proposal, audit, {"slots": all_slots}


def test_exact_never_claimed_scope(tmp_path):
    p, a, i = scope_fixture(tmp_path)
    assert len(continuation.select_slots(tmp_path, p, a, i)) == 138


@pytest.mark.parametrize(
    "mutation",
    [
        "include_old",
        "drop_new",
        "duplicate",
        "counts",
        "omitted_exclusion",
        "new_claim",
        "result_removed",
    ],
)
def test_cannot_repeat_or_expand_scope(tmp_path, mutation):
    p, a, i = scope_fixture(tmp_path)
    if mutation == "include_old":
        p["slots"][0] = i["slots"][0]
    elif mutation == "drop_new":
        p["slots"].pop()
    elif mutation == "duplicate":
        p["slots"][1] = p["slots"][0]
    elif mutation == "counts":
        p["counts"]["random_success"] = 59
    elif mutation == "omitted_exclusion":
        p["excluded_prior_claims"].pop()
    elif mutation == "new_claim":
        f = tmp_path / "short_success/attempts/task00__00.json"
        f.parent.mkdir(parents=True)
        f.write_text('{"task_name":"task00","sample_index":0}')
    else:
        f = tmp_path / "random_success/results.jsonl"
        f.write_text("\n".join(f.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        continuation.select_slots(tmp_path, p, a, i)


def parse_fixture(tmp_path):
    folder = tmp_path / "rollouts/task__00"
    (folder / "sampling").mkdir(parents=True)
    turns = [([10, 11], [12, 13]), ([10, 11, 12, 13, 14], [15]), ([30], [31, 32])]
    total = 0
    for n, (prompt, action) in enumerate(turns):
        (folder / f"sampling/{n:03}.request.json").write_text(
            json.dumps({"input_tokens": prompt, "max_tokens": 65536 - total})
        )
        (folder / f"sampling/{n:03}.response.json").write_text(
            json.dumps({"tokens": action, "logprobs": [-0.1] * len(action)})
        )
        total += len(action)
    trajectory = {
        "task_name": "task",
        "sample_index": 0,
        "stop_reason": "parse_error",
        "reward": 0,
        "turns": 3,
        "sampled_tokens": 5,
        "datums": [
            {
                "input_tokens": [10, 11, 12, 13, 14],
                "target_tokens": [11, 12, 13, 14, 15],
                "weights": [0, 1, 1, 0, 1],
            },
            {"input_tokens": [30, 31], "target_tokens": [31, 32], "weights": [1, 1]},
        ],
    }
    (folder / "trajectory.json").write_text(json.dumps(trajectory))
    result = TaskResult("task", 0, 0, {"parse_error": 1, "stop/parse_error": 1}, 3, 1)
    return folder, result


def test_real_parse_stop_needs_no_fabricated_messages(tmp_path):
    folder, result = parse_fixture(tmp_path)
    continuation.parse_stop_guard(tmp_path, ("task", 0), result)
    assert not (folder / "messages.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_response",
        "response_tokens",
        "response_float",
        "logprobs",
        "request_budget",
        "wrong_slot",
        "stop",
        "reward",
        "mask",
        "datum_float",
        "sampled_total",
        "incomplete_candidate",
    ],
)
def test_parse_error_acceptance_requires_exact_raw_provenance(tmp_path, mutation):
    folder, result = parse_fixture(tmp_path)
    path = folder / "trajectory.json"
    row = json.loads(path.read_text())
    if mutation == "missing_response":
        (folder / "sampling/002.response.json").unlink()
    elif mutation in ("response_tokens", "response_float", "logprobs"):
        p = folder / "sampling/002.response.json"
        r = json.loads(p.read_text())
        if mutation == "response_tokens":
            r["tokens"][0] = 99
        elif mutation == "response_float":
            r["tokens"][0] = 31.0
        else:
            r["logprobs"] = []
        p.write_text(json.dumps(r))
    elif mutation == "request_budget":
        p = folder / "sampling/001.request.json"
        r = json.loads(p.read_text())
        r["max_tokens"] = 65536
        p.write_text(json.dumps(r))
    elif mutation == "wrong_slot":
        row["task_name"] = "other"
    elif mutation == "stop":
        row["stop_reason"] = "completed"
    elif mutation == "reward":
        row["reward"] = 1
    elif mutation == "mask":
        row["datums"][0]["weights"][0] = 1
    elif mutation == "datum_float":
        row["datums"][0]["input_tokens"][0] = 10.0
    elif mutation == "sampled_total":
        row["sampled_tokens"] = 9
    else:
        (folder / "candidate.json").write_text('{"complete":false}')
    path.write_text(json.dumps(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        continuation.parse_stop_guard(tmp_path, ("task", 0), result)


def test_review_cannot_change_budget_or_identity(tmp_path):
    p = tmp_path / "review.json"
    identity = {"checkpoint": "one"}
    p.write_text(
        json.dumps(
            {
                "status": "accepted_never_claimed_continuation",
                "identity_sha256": continuation.digest(identity),
                "proposal_sha256": continuation.PROPOSAL_SHA,
                "maximum_new_rollouts": 138,
                "retry_permitted": False,
            }
        )
    )
    continuation.validate_review(str(p), identity)
    with pytest.raises(ValueError):
        continuation.validate_review(str(p), {"checkpoint": "two"})
    with pytest.raises(ValueError):
        continuation.validate_review(None, identity)


@pytest.mark.parametrize("kind", ["attempts", "terminal", "rollouts"])
def test_old_claim_artifact_in_new_output_rejected(tmp_path, kind):
    p = tmp_path / "random_success" / kind / ("old__00" if kind == "rollouts" else "old__00.json")
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(ValueError):
        continuation.check_prior(tmp_path, {"random_success": [("new", 0)], "short_success": []})


def test_blocked_before_any_claim_cannot_restart(tmp_path):
    (tmp_path / "status.json").write_text('{"stage":"blocked"}')
    with pytest.raises(ValueError):
        continuation.check_prior(tmp_path, {"random_success": [], "short_success": []})


@pytest.mark.asyncio
async def test_stop_and_drain_reused_dispatch(tmp_path):
    continuation.exclusive_json(tmp_path / "identity.json", {})
    event = asyncio.Event()
    calls = []

    async def operation(pair):
        calls.append(pair[1])
        if pair[1] == 0:
            await event.wait()
            raise ValueError("guard")
        event.set()
        await asyncio.sleep(0.01)
        calls.append("drained")
        return TaskResult("t", 1, 0, {}, 1, 1)

    results = await continuation.original.dispatch_slots(
        tmp_path, [("t", i) for i in range(4)], operation, 2
    )
    assert calls == [0, 1, "drained"] and results[2:] == [None, None]


@pytest.mark.parametrize("change", ["status", "proposal", "audit", "count", "exclusions"])
def test_independent_scope_receipt_must_match(tmp_path, change):
    row = {
        "status": "accepted_never_claimed_scope_only",
        "proposal_sha256": continuation.PROPOSAL_SHA,
        "blocked_audit_sha256": continuation.AUDIT_SHA,
        "maximum_new_rollouts": 138,
        "excluded_all_prior_claims": 22,
    }
    p = tmp_path / "review.json"
    p.write_text(json.dumps(row))
    continuation.scope_review(p)
    key = {
        "status": "status",
        "proposal": "proposal_sha256",
        "audit": "blocked_audit_sha256",
        "count": "maximum_new_rollouts",
        "exclusions": "excluded_all_prior_claims",
    }[change]
    row[key] = "changed"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        continuation.scope_review(p)


@pytest.mark.asyncio
async def test_exception_does_not_mutate_unrelated_phase(tmp_path, monkeypatch):
    (tmp_path / "identity.json").write_text('{"phase_kind":"other"}')
    (tmp_path / "status.json").write_text('{"stage":"immutable"}')

    async def failed(config):
        raise ValueError("identity mismatch")

    monkeypatch.setattr(continuation, "run", failed)
    cfg = continuation.Config(
        original_dir="old", output_dir=str(tmp_path), proposal_path="proposal", dispatch=True
    )
    with pytest.raises(ValueError):
        await continuation.main(cfg)
    assert json.loads((tmp_path / "status.json").read_text()) == {"stage": "immutable"}
