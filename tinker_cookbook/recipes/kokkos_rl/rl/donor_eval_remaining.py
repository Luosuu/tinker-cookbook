"""Continue the unclaimed difference of an explicitly reviewed evaluation history."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import chz
import tinker

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval as original
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval_continuation as continuation
from tinker_cookbook.recipes.kokkos_rl.rl.donor_qualification import records
from tinker_cookbook.recipes.kokkos_rl.rl.saved_candidate_regrade import digest, exclusive_json
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json


@chz.chz
class Config:
    proposal_path: str
    proposal_sha256: str
    scope_review_path: str
    output_dir: str
    dispatch: bool = False
    review_receipt: str | None = None
    env_file: str = ".env"


def key(row: dict[str, object]) -> tuple[str, str, int]:
    arm, task, index = row.get("arm"), row.get("task_name"), row.get("sample_index")
    if (
        arm not in original.ARMS
        or not isinstance(task, str)
        or type(index) is not int
        or not 0 <= index < 4
    ):
        raise ValueError("Malformed evaluation slot")
    return str(arm), task, index


def phase_inventory(
    directory: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, object]], list[dict[str, str]]]:
    claimed = {}
    proofs = []
    identity_path = directory / "identity.json"
    proofs.append(original.pinned_file(identity_path))
    proofs.append(original.pinned_file(directory / "status.json"))
    for arm in original.ARMS:
        folder = directory / arm
        rows = original.rows_by_pair(folder / "results.jsonl")
        if (folder / "results.jsonl").exists():
            proofs.append(original.pinned_file(folder / "results.jsonl"))
        pairs = set()
        for path in sorted((folder / "attempts").glob("*.json")):
            claim = read_json(path)
            slot = key(
                {
                    "arm": arm,
                    "task_name": claim.get("task_name"),
                    "sample_index": claim.get("sample_index"),
                }
            )
            if path.stem != f"{slot[1]}__{slot[2]:02d}" or slot in claimed:
                raise ValueError("Ambiguous historical claim")
            pairs.add(slot[1:])
            claimed[slot] = claim
            proofs.append(original.pinned_file(path))
        if pairs != set(rows):
            raise ValueError("Historical partial/unknown result requires review before continuing")
        for path in (folder / "rollouts").glob("*"):
            if path.name not in {f"{t}__{i:02d}" for t, i in pairs}:
                raise ValueError("Unclaimed historical rollout artifact")
        for path in (folder / "terminal").glob("*.json"):
            row = read_json(path)
            slot = key(
                {
                    "arm": arm,
                    "task_name": row.get("task_name"),
                    "sample_index": row.get("sample_index"),
                }
            )
            if slot not in claimed or path.stem != f"{slot[1]}__{slot[2]:02d}":
                raise ValueError("Unknown historical terminal")
            if row != cast(dict[str, object], asdict(rows[slot[1:]])):
                raise ValueError("Historical terminal/result differ")
            proofs.append(original.pinned_file(path))
    return claimed, proofs


def selection(
    proposal: dict[str, object],
    full: list[dict[str, object]],
    inventories: list[dict[tuple[str, str, int], dict[str, object]]],
) -> list[dict[str, object]]:
    universe = {key(s) for s in full}
    if len(universe) != len(full) or len(full) != 160:
        raise ValueError("Original budget must contain exactly160 unique slots")
    if {a: sum(s[0] == a for s in universe) for a in original.ARMS} != dict.fromkeys(
        original.ARMS, 80
    ):
        raise ValueError("Original budget must have80 slots per arm")
    claimed = set()
    for inventory in inventories:
        if claimed & set(inventory) or not set(inventory) <= universe:
            raise ValueError("Duplicate or out-of-budget historical claim")
        claimed.update(inventory)
    pending = [s for s in full if key(s) not in claimed]
    if (
        proposal.get("total_budget_slots") != 160
        or proposal.get("excluded_claims") != len(claimed)
        or proposal.get("maximum_new_rollouts") != len(pending)
        or proposal.get("slots") != pending
        or proposal.get("counts") != {a: sum(s["arm"] == a for s in pending) for a in original.ARMS}
        or not pending
    ):
        raise ValueError("Proposal is not exact historical-claims difference")
    return pending


def verify_archive(audit_path: Path) -> dict[str, object]:
    audit = read_json(audit_path)
    for source, value in mapping(audit["proofs"]).items():
        proof = mapping(value)
        path = Path(source)
        raw = path.read_bytes()
        obj = (audit_path.parent / str(proof["object"])).resolve()
        if not obj.is_relative_to((audit_path.parent / "objects").resolve()):
            raise ValueError("Frozen object escapes archive")
        if (
            len(raw) != proof["bytes"]
            or original.pinned_file(path)["sha256"] != proof["sha256"]
            or obj.read_bytes() != raw
        ):
            raise ValueError("Historical source/object changed")
    return audit


def scope_review(path: Path, proposal_sha: str, pending: int, excluded: int) -> dict[str, str]:
    row = read_json(path)
    if (
        row.get("status") != "accepted_remaining_eval_scope_only"
        or row.get("proposal_sha256") != proposal_sha
        or row.get("remaining_slots") != pending
        or row.get("excluded_claims") != excluded
        or row.get("total_slots") != 160
    ):
        raise ValueError("Exact multi-phase scope review required")
    return original.pinned_file(path)


def review(path: str | None, identity: dict[str, object]) -> None:
    if path is None or read_json(Path(path)) != {
        "status": "accepted_remaining_eval",
        "identity_sha256": digest(identity),
        "maximum_new_rollouts": identity["maximum_new_rollouts"],
        "total_budget_slots": 160,
        "retry_permitted": False,
    }:
        raise ValueError("Exact remaining evaluation identity review required")


def capacity() -> dict[str, object]:
    snapshot = continuation.capacity()
    for line in subprocess.check_output(
        ["ps", "-axo", "pid=,stat=,command="], text=True
    ).splitlines():
        if (
            "-m tinker_cookbook.recipes.kokkos_rl.rl.donor_eval_remaining " in line
            and int(line.strip().split(maxsplit=2)[0]) != os.getpid()
        ):
            raise ValueError("Another remaining-evaluation controller is active")
    return snapshot


async def run(config: Config) -> None:
    proposal_path = Path(config.proposal_path).resolve()
    output = Path(config.output_dir).resolve()
    if original.pinned_file(proposal_path)["sha256"] != config.proposal_sha256:
        raise ValueError("Proposal changed")
    proposal = read_json(proposal_path)
    first_path = continuation.checked(mapping(proposal["original_identity"]))
    first = read_json(first_path)
    first_dir = first_path.parent
    full = records(first["slots"])
    phases = records(proposal["phases"])
    if not phases or Path(str(phases[0]["directory"])).resolve() != first_dir:
        raise ValueError("History must start with original fixed evaluation")
    inventories = []
    phase_records = []
    proofs = [original.pinned_file(proposal_path)]
    source_map = {}
    for entry in phases:
        directory = Path(str(entry["directory"])).resolve()
        if any(directory == Path(str(p["directory"])) for p in phase_records):
            raise ValueError("Repeated historical directory")
        if (
            output == directory
            or output.is_relative_to(directory)
            or directory.is_relative_to(output)
        ):
            raise ValueError("Output overlaps history")
        identity_path = continuation.checked(mapping(entry["identity"]))
        if identity_path != directory / "identity.json":
            raise ValueError("Phase identity path differs")
        identity = read_json(identity_path)
        audit_path = continuation.checked(mapping(entry["audit"]))
        audit = verify_archive(audit_path)
        inventory, phase_proofs = phase_inventory(directory)
        audited = {
            key(
                {
                    "arm": s["arm"],
                    "task_name": mapping(s["result"])["task_name"],
                    "sample_index": mapping(s["result"])["sample_index"],
                }
            )
            for s in records(audit["slots"])
        }
        if set(inventory) != audited or len(inventory) != entry["expected_claims"]:
            raise ValueError("Audited historical scope differs")
        if (
            identity["checkpoints"] != first["checkpoints"]
            or identity["qualified"] != first["qualified"]
            or identity["versions"] != first["versions"]
        ):
            raise ValueError("Historical evaluation protocol differs")
        for arm in original.ARMS:
            old_cfg = mapping(mapping(identity["configs"])[arm])
            reference = mapping(mapping(first["configs"])[arm])
            if {k: v for k, v in old_cfg.items() if k not in ("output_path", "resume_dir")} != {
                k: v for k, v in reference.items() if k not in ("output_path", "resume_dir")
            }:
                raise ValueError("Historical model/budget config changed")
        for source in records(identity["execution_sources"]):
            if source["path"] in source_map and source_map[source["path"]] != source:
                raise ValueError("Historical source versions differ")
            source_map[str(source["path"])] = source
        inventories.append(inventory)
        proofs.extend(phase_proofs)
        proofs.append(original.pinned_file(audit_path))
        phase_records.append(
            {
                "directory": str(directory),
                "identity": original.pinned_file(identity_path),
                "audit": original.pinned_file(audit_path),
                "claims": [list(k) for k in sorted(inventory)],
            }
        )
    slots = selection(proposal, full, inventories)
    accepted_scope = scope_review(
        Path(config.scope_review_path).resolve(),
        config.proposal_sha256,
        len(slots),
        160 - len(slots),
    )
    proofs.append(accepted_scope)
    collection = Path(str(mapping(first["original_manifest"])["path"])).parent
    if (
        output == collection
        or output.is_relative_to(collection)
        or collection.is_relative_to(output)
    ):
        raise ValueError("Output overlaps collection")
    manifest = read_json(continuation.checked(mapping(first["original_manifest"])))
    original_config = read_json(continuation.checked(mapping(first["original_config"])))
    for name in ("manifest.json", "config.json"):
        proofs.append(original.pinned_file(collection / name))
    tasks = original.load_harbor_tasks_from_dir(collection / "tasks")
    if {t.task_name: original.current_task_digest(t) for t in tasks} != manifest["task_hashes"]:
        raise ValueError("Frozen task content changed")
    plan = read_json(continuation.checked(mapping(first["plan"])))
    if original.expected_slots(manifest, plan) != full:
        raise ValueError("Original fixed split/budget changed")
    qualified = original.qualified_evidence(
        tasks, Path(str(original_config["qualified_bundle_dir"]))
    )
    if (
        qualified != first["qualified"]
        or proposal["qualified"] != qualified
        or proposal["checkpoints"] != first["checkpoints"]
    ):
        raise ValueError("Qualification or checkpoints changed")
    training_proofs = [cast(dict[str, str], p) for p in records(first["training_proofs"])]
    original.check_proofs(training_proofs)
    proofs.extend(training_proofs)
    versions = {n: importlib.metadata.version(n) for n in ("tinker", "tml-renderers", "chz")}
    if versions != first["versions"]:
        raise ValueError("Dependency versions changed")
    for path in (Path(continuation.__file__).resolve(), Path(__file__).resolve()):
        proof = original.pinned_file(path)
        if str(path) in source_map and source_map[str(path)] != proof:
            raise ValueError("Previously pinned source changed")
        source_map[str(path)] = proof
    sources = [cast(dict[str, str], p) for _, p in sorted(source_map.items())]
    original.check_proofs(sources)
    configs = {}
    for arm in original.ARMS:
        cfg = chz.replace(
            original.eval_config(first_dir / arm, original.MODEL, qualified, 4),
            checkpoint_url=str(mapping(first["checkpoints"])[arm]),
        )
        original.validate_budget(plan, cfg)
        if chz.asdict(cfg) != mapping(first["configs"])[arm]:
            raise ValueError("Actual runtime config differs")
        configs[arm] = chz.replace(cfg, output_path=str(output / arm), resume_dir=str(output / arm))
    identity = {
        "phase_kind": "donor_eval_remaining_v1",
        "proposal": original.pinned_file(proposal_path),
        "scope_review": accepted_scope,
        "history": phase_records,
        "input_proofs": proofs,
        "execution_sources": sources,
        "versions": versions,
        "slots": slots,
        "excluded_claims": 160 - len(slots),
        "maximum_new_rollouts": len(slots),
        "total_budget_slots": 160,
        "checkpoints": first["checkpoints"],
        "qualified": qualified,
        "configs": {a: chz.asdict(c) for a, c in configs.items()},
        "capacity_policy": original.CAPACITY_POLICY,
        "guard": "exact_raw_parse_error_v1",
        "retry_permitted": False,
        "automatic_training": False,
        "old_results_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "identity.json").exists():
            if read_json(output / "identity.json") != identity:
                raise ValueError("Remaining phase identity changed")
        else:
            exclusive_json(output / "identity.json", identity)
        pairs = {a: [(s[1], s[2]) for s in map(key, slots) if s[0] == a] for a in original.ARMS}
        for arm in original.ARMS:
            arm_identity = {
                "phase": original.pinned_file(output / "identity.json"),
                "arm": arm,
                "checkpoint": mapping(first["checkpoints"])[arm],
            }
            p = output / arm / "identity.json"
            if p.exists():
                if read_json(p) != arm_identity:
                    raise ValueError("Arm identity changed")
            else:
                exclusive_json(p, arm_identity)
        pending = continuation.check_prior(output, pairs)
        preflight = {
            "identity_sha256": digest(identity),
            "pending": {a: len(p) for a, p in pending.items()},
            "excluded_claims": 160 - len(slots),
            "maximum_new_rollouts": len(slots),
            "total_budget_slots": 160,
            "dispatch": config.dispatch,
            "model_requests": 0,
            "sandbox_requests": 0,
        }
        original.write_json(output / "preflight.json", preflight)
        print(json.dumps(preflight), flush=True)
        if not config.dispatch:
            return
        review(config.review_receipt, identity)
        if not any(pending.values()):
            return
        original.write_json(output / "capacity_startup.json", capacity())
        original.load_env_file(Path(config.env_file))
        cache = output / "contree_images.json"
        if not cache.exists():
            shutil.copyfile(Path(str(phases[-1]["directory"])) / "contree_images.json", cache)
        primary = original.ContreeDockerfileSandboxFactory(
            cache, timeout=3600, runtime_build_parallelism=1, allow_network=False
        )
        primary._client = original.create_polling_client(
            primary._client.config, original.OperationPollPolicy()
        )
        factory = original.qualified_factory(
            primary, tasks, str(qualified["resource_policy_version"])
        )
        tokenizer = original.tokenizer_utils.get_tokenizer(original.MODEL)
        renderer = original.get_renderer(
            original.model_info.get_recommended_renderer_name(original.MODEL), tokenizer
        )
        by_name = {t.task_name: t for t in tasks}
        original.write_json(
            output / "process.json",
            {
                "pid": os.getpid(),
                "at": datetime.now(UTC).isoformat(),
                "identity_sha256": digest(identity),
            },
        )

        def guard() -> None:
            original.check_proofs(sources)
            original.check_proofs(proofs)
            current = [phase_inventory(Path(str(p["directory"])))[0] for p in phases]
            if selection(proposal, full, current) != slots:
                raise ValueError("Historical claims changed before dispatch")

        for arm in original.ARMS:
            chosen = pending[arm]
            if not chosen:
                continue
            guard()
            original.write_json(output / f"capacity_{arm}.json", capacity())
            if (
                original.qualified_evidence(
                    tasks, Path(str(original_config["qualified_bundle_dir"]))
                )
                != qualified
            ):
                raise ValueError("Qualified evidence changed")
            folder, cfg = output / arm, configs[arm]
            original.prepare_eval_state(
                folder,
                chz.asdict(cfg),
                [t for t in tasks if t.task_name in {p[0] for p in pairs[arm]}],
                evaluator="tinker-harbor",
            )
            original.write_json(folder / "config.json", chz.asdict(cfg))
            service = tinker.ServiceClient()
            sampler = await service.create_sampling_client_async(
                model_path=cfg.checkpoint_url, base_model=original.MODEL
            )
            policy = original.TinkerTokenCompleter(
                sampler, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )
            result_lock = asyncio.Lock()
            original.write_json(
                output / "status.json", {"stage": "evaluating", "arm": arm, "pending": len(chosen)}
            )

            async def operation(pair: tuple[str, int]) -> TaskResult:
                guard()
                task = by_name[pair[0]]
                if original.current_task_digest(task) != mapping(manifest["task_hashes"])[pair[0]]:
                    raise ValueError("Task changed before dispatch")
                result = await original.evaluate_task(
                    task, policy, renderer, factory, cfg, folder, result_lock, tokenizer, pair[1]
                )
                continuation.parse_stop_guard(folder, pair, result)
                return result

            completed = await original.dispatch_slots(folder, chosen, operation, 4)
            errors = [f"{type(r).__name__}: {r}" for r in completed if isinstance(r, BaseException)]
            terminal = original.rows_by_pair(folder / "results.jsonl")
            if (
                errors
                or any(r.error is not None for r in terminal.values())
                or not set(chosen).issubset(terminal)
            ):
                original.write_json(
                    output / "status.json",
                    {
                        "stage": "blocked",
                        "arm": arm,
                        "errors": errors,
                        "terminal": len(terminal),
                        "automatic_restart": False,
                    },
                )
                return
            original.write_json(
                folder / "efficiency.json",
                original.delivery_metrics(list(terminal.values()), folder),
            )
        original.write_json(
            output / "status.json",
            {
                "stage": "awaiting_evaluation_review",
                "terminal": {
                    a: len(original.rows_by_pair(output / a / "results.jsonl"))
                    for a in original.ARMS
                },
                "old_results_modified": False,
                "automatic_training": False,
            },
        )


async def main(config: Config) -> None:
    try:
        await run(config)
    except BaseException as error:
        output = Path(config.output_dir).resolve()
        if (
            config.dispatch
            and (output / "identity.json").exists()
            and read_json(output / "identity.json").get("phase_kind") == "donor_eval_remaining_v1"
        ):
            failure = {
                "stage": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "at": datetime.now(UTC).isoformat(),
                "automatic_restart": False,
            }
            exclusive_json(
                output
                / "controller_failures"
                / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
                failure,
            )
            with (output / "controller.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    original.write_json(output / "status.json", failure)
        raise


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
