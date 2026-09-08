"""Reviewed multi-phase set difference, immutable proof and no-repeat boundaries."""

import json
from dataclasses import asdict

import pytest

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval_remaining as remaining


def selection_fixture():
    full = [
        {"arm": arm, "task_name": f"t{i:02d}", "sample_index": n, "task_sha256": "hash"}
        for arm in remaining.original.ARMS
        for n in range(4)
        for i in range(20)
    ]
    inventories = [
        {remaining.key(s): {} for s in full[:22]},
        {remaining.key(s): {} for s in full[22:68]},
    ]
    proposal = {
        "total_budget_slots": 160,
        "excluded_claims": 68,
        "maximum_new_rollouts": 92,
        "slots": full[68:],
        "counts": {"random_success": 12, "short_success": 80},
    }
    return proposal, full, inventories


def test_two_histories_give_exact92():
    p, f, i = selection_fixture()
    assert remaining.selection(p, f, i) == f[68:]


def test_third_history_requires_no_code_change():
    p, f, i = selection_fixture()
    i.append({remaining.key(s): {} for s in f[68:72]})
    p.update(
        excluded_claims=72,
        maximum_new_rollouts=88,
        slots=f[72:],
        counts={"random_success": 8, "short_success": 80},
    )
    assert len(remaining.selection(p, f, i)) == 88


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "outside",
        "duplicate_universe",
        "expand160",
        "wrong_pending",
        "missing_history",
        "wrong_count",
        "reorder",
    ],
)
def test_history_or_budget_ambiguity_rejected(change):
    p, f, i = selection_fixture()
    if change == "overlap":
        i[1][next(iter(i[0]))] = {}
    elif change == "outside":
        i[1][("random_success", "outside", 0)] = {}
    elif change == "duplicate_universe":
        f[1] = f[0]
    elif change == "expand160":
        f.append({"arm": "random_success", "task_name": "extra", "sample_index": 0})
    elif change == "wrong_pending":
        p["slots"] = [f[0], *p["slots"][1:]]
    elif change == "missing_history":
        i.pop()
    elif change == "wrong_count":
        p["excluded_claims"] = 67
    else:
        p["slots"] = list(reversed(p["slots"]))
    with pytest.raises(ValueError):
        remaining.selection(p, f, i)


@pytest.mark.parametrize(
    "row",
    [
        {"arm": "other", "task_name": "t", "sample_index": 0},
        {"arm": "random_success", "task_name": "t", "sample_index": False},
        {"arm": "short_success", "task_name": "t", "sample_index": 4},
    ],
)
def test_slot_types_are_strict(row):
    with pytest.raises(ValueError):
        remaining.key(row)


def phase_fixture(tmp_path):
    (tmp_path / "identity.json").write_text("{}")
    (tmp_path / "status.json").write_text('{"stage":"blocked"}')
    folder = tmp_path / "random_success"
    (folder / "attempts").mkdir(parents=True)
    (folder / "terminal").mkdir()
    row = TaskResult("t", 0, 0, {}, 40, 1, "TimeoutError")
    (folder / "attempts/t__00.json").write_text('{"task_name":"t","sample_index":0}')
    (folder / "results.jsonl").write_text(json.dumps(asdict(row)) + "\n")
    (folder / "terminal/t__00.json").write_text(json.dumps(asdict(row)))
    return folder


def test_failed_claim_is_still_excluded(tmp_path):
    phase_fixture(tmp_path)
    inventory, _ = remaining.phase_inventory(tmp_path)
    assert set(inventory) == {("random_success", "t", 0)}


@pytest.mark.parametrize(
    "change",
    ["claim_without_result", "result_without_claim", "unknown_partial", "terminal_mismatch"],
)
def test_history_partial_or_mismatch_blocks(tmp_path, change):
    f = phase_fixture(tmp_path)
    if change == "claim_without_result":
        (f / "results.jsonl").write_text("")
    elif change == "result_without_claim":
        (f / "attempts/t__00.json").unlink()
    elif change == "unknown_partial":
        (f / "rollouts/other__00").mkdir(parents=True)
    else:
        p = f / "terminal/t__00.json"
        r = json.loads(p.read_text())
        r["reward"] = 1
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        remaining.phase_inventory(tmp_path)


def archive_fixture(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixed")
    proof = remaining.original.pinned_file(source)
    obj = tmp_path / "objects" / proof["sha256"]
    obj.parent.mkdir()
    obj.write_text("fixed")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "proofs": {
                    str(source): {
                        "sha256": proof["sha256"],
                        "bytes": 5,
                        "object": str(obj.relative_to(tmp_path)),
                    }
                }
            }
        )
    )
    return audit, source, obj


@pytest.mark.parametrize("change", ["source", "object", "escape"])
def test_frozen_audit_tampering_rejected(tmp_path, change):
    audit, source, obj = archive_fixture(tmp_path)
    remaining.verify_archive(audit)
    if change == "source":
        source.write_text("changed")
    elif change == "object":
        obj.write_text("wrong")
    else:
        r = json.loads(audit.read_text())
        r["proofs"][str(source)]["object"] = "source"
        audit.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        remaining.verify_archive(audit)


def test_exact_scope_review_and_dispatch_review(tmp_path):
    p = tmp_path / "scope.json"
    p.write_text(
        json.dumps(
            {
                "status": "accepted_remaining_eval_scope_only",
                "proposal_sha256": "abc",
                "remaining_slots": 92,
                "excluded_claims": 68,
                "total_slots": 160,
            }
        )
    )
    remaining.scope_review(p, "abc", 92, 68)
    with pytest.raises(ValueError):
        remaining.scope_review(p, "abc", 93, 67)
    identity = {"maximum_new_rollouts": 92, "slots": ["fixed"]}
    q = tmp_path / "dispatch.json"
    q.write_text(
        json.dumps(
            {
                "status": "accepted_remaining_eval",
                "identity_sha256": remaining.digest(identity),
                "maximum_new_rollouts": 92,
                "total_budget_slots": 160,
                "retry_permitted": False,
            }
        )
    )
    remaining.review(str(q), identity)
    with pytest.raises(ValueError):
        remaining.review(str(q), {"maximum_new_rollouts": 92, "slots": ["changed"]})
    with pytest.raises(ValueError):
        remaining.review(None, identity)


@pytest.mark.asyncio
async def test_outer_failure_preserved_and_never_retried(tmp_path, monkeypatch):
    (tmp_path / "identity.json").write_text('{"phase_kind":"donor_eval_remaining_v1"}')
    calls = []

    async def operation(config):
        calls.append(1)
        raise RuntimeError("client failure")

    monkeypatch.setattr(remaining, "run", operation)
    cfg = remaining.Config(
        proposal_path="p",
        proposal_sha256="s",
        scope_review_path="r",
        output_dir=str(tmp_path),
        dispatch=True,
    )
    with pytest.raises(RuntimeError):
        await remaining.main(cfg)
    assert calls == [1] and json.loads((tmp_path / "status.json").read_text())["stage"] == "blocked"


@pytest.mark.asyncio
async def test_lock_contender_cannot_clobber_live_owner(tmp_path, monkeypatch):
    import fcntl

    (tmp_path / "identity.json").write_text('{"phase_kind":"donor_eval_remaining_v1"}')
    (tmp_path / "status.json").write_text('{"stage":"evaluating"}')

    async def fail(config):
        raise BlockingIOError("lock")

    monkeypatch.setattr(remaining, "run", fail)
    cfg = remaining.Config(
        proposal_path="p",
        proposal_sha256="s",
        scope_review_path="r",
        output_dir=str(tmp_path),
        dispatch=True,
    )
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            await remaining.main(cfg)
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "evaluating"
