"""Evaluate only never-claimed slots after a separately reviewed blocked phase."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import chz
import tinker

from tinker_cookbook.recipes.harbor_rl.eval import TaskResult
from tinker_cookbook.recipes.kokkos_rl.rl import donor_eval as original
from tinker_cookbook.recipes.kokkos_rl.rl.donor_qualification import records
from tinker_cookbook.recipes.kokkos_rl.rl.saved_candidate_regrade import digest, exclusive_json
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json

PROPOSAL_SHA = "a0afc3529a3fd31f8c7dc46c927636a58f78f01bd3888f5b1a9efc79267afaf2"
AUDIT_SHA = "ac5ee2b868c31dc530fbf93cfb2540c713c605483f6e4d5bf0e76d070dc73d25"


@chz.chz
class Config:
    original_dir: str
    output_dir: str
    proposal_path: str
    dispatch: bool = False
    review_receipt: str | None = None
    env_file: str = ".env"


def scope_review(path: Path) -> dict[str, str]:
    review = read_json(path)
    if (
        review.get("status") != "accepted_never_claimed_scope_only"
        or review.get("blocked_audit_sha256") != AUDIT_SHA
        or review.get("proposal_sha256") != PROPOSAL_SHA
        or review.get("maximum_new_rollouts") != 138
        or review.get("excluded_all_prior_claims") != 22
    ):
        raise ValueError("Exact independently accepted continuation scope required")
    return original.pinned_file(path)


def checked(proof: dict[str, object]) -> Path:
    path = Path(str(proof["path"]))
    if original.pinned_file(path)["sha256"] != proof.get("sha256"):
        raise ValueError("Original evidence changed")
    return path


def select_slots(
    old: Path, proposal: dict[str, object], audit: dict[str, object], identity: dict[str, object]
) -> list[dict[str, object]]:
    claims = set()
    for arm in original.ARMS:
        results = original.rows_by_pair(old / arm / "results.jsonl")
        arm_claims = set()
        for p in (old / arm / "attempts").glob("*.json"):
            marker = read_json(p)
            pair = str(marker["task_name"]), int(str(marker["sample_index"]))
            if p.stem != f"{pair[0]}__{pair[1]:02d}" or pair in arm_claims:
                raise ValueError("Ambiguous original claim")
            arm_claims.add(pair)
            claims.add((arm, *pair))
        if arm_claims != set(results):
            raise ValueError("Original claimed/result scope changed")
    recorded = {
        (
            str(s["arm"]),
            str(mapping(s["result"])["task_name"]),
            int(str(mapping(s["result"])["sample_index"])),
        )
        for s in records(audit["slots"])
    }
    if claims != recorded or len(claims) != 22:
        raise ValueError("All original22 claims must remain excluded")
    expected = [
        s
        for s in records(identity["slots"])
        if (str(s["arm"]), str(s["task_name"]), int(str(s["sample_index"]))) not in claims
    ]
    excluded = [
        {"arm": arm, "task_name": task, "sample_index": index}
        for arm, task, index in sorted(claims)
    ]
    if (
        proposal.get("slots") != expected
        or proposal.get("excluded_prior_claims") != excluded
        or proposal.get("maximum_new_rollouts") != 138
        or len(expected) != 138
        or proposal.get("counts") != {"random_success": 58, "short_success": 80}
    ):
        raise ValueError("Continuation must contain exactly58+80 never-claimed slots")
    if len({(s["arm"], s["task_name"], s["sample_index"]) for s in expected}) != 138:
        raise ValueError("Duplicate continuation slot")
    return expected


def parse_stop_guard(folder: Path, pair: tuple[str, int], result: TaskResult) -> None:
    """Accept a real model parse failure from exact raw tokens, without invented history."""
    if result.error is not None:
        return
    rollout = folder / "rollouts" / f"{pair[0]}__{pair[1]:02d}"
    if (rollout / "candidate.json").exists():
        original.check_candidate_or_model_stop(folder, pair, result)
        return
    row = read_json(rollout / "trajectory.json")
    if (
        row.get("stop_reason") != "parse_error"
        or row.get("task_name") != pair[0]
        or row.get("sample_index") != pair[1]
        or row.get("reward") != 0
        or result.reward != 0
        or row.get("turns") != result.turns_used
        or result.reward_details.get("parse_error") != 1
        or result.reward_details.get("stop/parse_error") != 1
        or not 0 < result.turns_used <= 40
    ):
        raise ValueError("Missing candidate lacks genuine parse-error evidence")
    names = [f"{i:03}" for i in range(result.turns_used)]
    requests = sorted((rollout / "sampling").glob("*.request.json"))
    responses = sorted((rollout / "sampling").glob("*.response.json"))
    if [p.name.split(".")[0] for p in requests] != names or [
        p.name.split(".")[0] for p in responses
    ] != names:
        raise ValueError("Parse-error raw request/response set incomplete")
    groups = []
    seq: list[int] = []
    mask: list[int] = []
    total = 0
    for request, response in zip(requests, responses, strict=True):
        req, res = read_json(request), read_json(response)
        prompt, action = req.get("input_tokens"), res.get("tokens")
        if (
            not isinstance(prompt, list)
            or not isinstance(action, list)
            or not all(type(t) is int and t >= 0 for t in prompt + action)
        ):
            raise ValueError("Raw tokens are not integers")
        if req.get("max_tokens") != 65536 - total or not 0 < len(action) <= min(
            16384, 65536 - total
        ):
            raise ValueError("Raw sampling budget differs")
        probs = res.get("logprobs")
        if (
            not isinstance(probs, list)
            or len(probs) != len(action)
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in probs)
        ):
            raise ValueError("Raw logprobs incomplete")
        if seq and prompt[: len(seq)] != seq:
            groups.append((seq, mask))
            seq, mask = [], []
        mask.extend([0] * (len(prompt) - len(seq)))
        seq = list(prompt)
        seq.extend(action)
        mask.extend([1] * len(action))
        total += len(action)
    groups.append((seq, mask))
    datums = records(row["datums"])
    if len(datums) != len(groups) or row.get("sampled_tokens") != total or total > 65536:
        raise ValueError("Parse trajectory disagrees with raw response totals")
    for datum, (tokens, weights) in zip(datums, groups, strict=True):
        for field in ("input_tokens", "target_tokens"):
            values = datum.get(field)
            if not isinstance(values, list) or not all(type(t) is int and t >= 0 for t in values):
                raise ValueError("Trajectory tokens are not integers")
        values = datum.get("weights")
        if not isinstance(values, list) or not all(
            type(w) in (float, int) and w in (0, 1) for w in values
        ):
            raise ValueError("Trajectory mask is not binary")
        if (
            len(tokens) - 1 > 114688
            or datum.get("input_tokens") != tokens[:-1]
            or datum.get("target_tokens") != tokens[1:]
            or datum.get("weights") != weights[1:]
        ):
            raise ValueError("Parse trajectory raw tokens/shift/mask differ")


def capacity() -> dict[str, object]:
    result = original.capacity_snapshot()
    listing = subprocess.check_output(["ps", "-axo", "pid=,stat=,command="], text=True)
    for line in listing.splitlines():
        if (
            "-m tinker_cookbook.recipes.kokkos_rl.rl.donor_eval_continuation " in line
            and int(line.strip().split(maxsplit=2)[0]) != os.getpid()
        ):
            raise ValueError("Another continuation occupies shared capacity")
    return result


def check_prior(
    output: Path, pairs: dict[str, list[tuple[str, int]]]
) -> dict[str, list[tuple[str, int]]]:
    path = output / "status.json"
    if path.exists() and read_json(path).get("stage") == "blocked":
        raise ValueError("Blocked continuation requires explicit recovery")
    for arm, selected in pairs.items():
        names = {f"{task}__{index:02d}" for task, index in selected}
        for kind in ("attempts", "terminal", "rollouts"):
            for p in (output / arm / kind).glob("*"):
                if (p.name if kind == "rollouts" else p.stem) not in names:
                    raise ValueError("Artifact outside never-claimed continuation scope")
    return {a: original.pending_slots(output / a, p) for a, p in pairs.items()}


def validate_review(path: str | None, identity: dict[str, object]) -> None:
    if path is None or read_json(Path(path)) != {
        "status": "accepted_never_claimed_continuation",
        "identity_sha256": digest(identity),
        "proposal_sha256": PROPOSAL_SHA,
        "maximum_new_rollouts": 138,
        "retry_permitted": False,
    }:
        raise ValueError("Exact continuation identity review required")


async def run(config: Config) -> None:
    old, output = Path(config.original_dir).resolve(), Path(config.output_dir).resolve()
    if output == old or output.is_relative_to(old) or old.is_relative_to(output):
        raise ValueError("Use an independent continuation directory")
    proposal_path = Path(config.proposal_path).resolve()
    if original.pinned_file(proposal_path)["sha256"] != PROPOSAL_SHA:
        raise ValueError("Unreviewed proposal")
    proposal = read_json(proposal_path)
    audit_path = checked(mapping(proposal["blocked_audit"]))
    if original.pinned_file(audit_path)["sha256"] != AUDIT_SHA:
        raise ValueError("Wrong blocked audit")
    audit = read_json(audit_path)
    for source, value in mapping(audit["proofs"]).items():
        proof = mapping(value)
        raw = Path(source).read_bytes()
        obj = audit_path.parent / str(proof["object"])
        if (
            len(raw) != proof["bytes"]
            or original.pinned_file(Path(source))["sha256"] != proof["sha256"]
            or obj.read_bytes() != raw
        ):
            raise ValueError("Frozen original audit source/object changed")
    original_identity = read_json(checked(mapping(proposal["original_identity"])))
    if Path(str(mapping(proposal["original_identity"])["path"])).parent != old:
        raise ValueError("Original directory differs")
    checked(mapping(proposal["original_results"]))
    if read_json(old / "status.json").get("stage") != "blocked":
        raise ValueError("Original stopped state changed")
    accepted_scope = scope_review(old / "monitor/root_never_claimed_scope_review.json")
    slots = select_slots(old, proposal, audit, original_identity)
    collection = Path(str(mapping(original_identity["original_manifest"])["path"])).parent
    if (
        output == collection
        or output.is_relative_to(collection)
        or collection.is_relative_to(output)
    ):
        raise ValueError("Output overlaps source collection")
    manifest = read_json(collection / "manifest.json")
    original_config = read_json(collection / "config.json")
    original.check_proofs(
        [cast(dict[str, str], p) for p in records(original_identity["execution_sources"])]
    )
    original.check_proofs(
        [cast(dict[str, str], p) for p in records(original_identity["training_proofs"])]
    )
    tasks = original.load_harbor_tasks_from_dir(collection / "tasks")
    if {t.task_name: original.current_task_digest(t) for t in tasks} != manifest["task_hashes"]:
        raise ValueError("Task bytes changed")
    qualified = original.qualified_evidence(
        tasks, Path(str(original_config["qualified_bundle_dir"]))
    )
    if (
        qualified != original_identity["qualified"]
        or proposal["qualification"] != qualified
        or proposal["checkpoints"] != original_identity["checkpoints"]
    ):
        raise ValueError("Checkpoint or qualification changed")
    original_plan = read_json(checked(mapping(original_identity["plan"])))
    configs = {}
    for arm in original.ARMS:
        cfg = original.eval_config(old / arm, original.MODEL, qualified, 4)
        cfg = chz.replace(cfg, checkpoint_url=str(mapping(original_identity["checkpoints"])[arm]))
        if chz.asdict(cfg) != mapping(original_identity["configs"])[arm]:
            raise ValueError("Continuation defaults differ from exact original config")
        original.validate_budget(original_plan, cfg)
        configs[arm] = chz.replace(cfg, output_path=str(output / arm), resume_dir=str(output / arm))
    sources = [
        *records(original_identity["execution_sources"]),
        original.pinned_file(Path(__file__).resolve()),
    ]
    old_proofs = [
        original.pinned_file(p)
        for p in [
            old / "identity.json",
            old / "status.json",
            old / "random_success/results.jsonl",
            proposal_path,
            audit_path,
            collection / "manifest.json",
            collection / "config.json",
            *sorted((old / "random_success/attempts").glob("*.json")),
        ]
    ]
    old_proofs.append(accepted_scope)
    versions = {
        name: importlib.metadata.version(name) for name in ("tinker", "tml-renderers", "chz")
    }
    if versions != original_identity["versions"]:
        raise ValueError("Dependency version changed")
    identity = {
        "phase_kind": "donor_eval_never_claimed_v1",
        "original": original.pinned_file(old / "identity.json"),
        "proposal": original.pinned_file(proposal_path),
        "blocked_audit": original.pinned_file(audit_path),
        "accepted_scope": accepted_scope,
        "original_proofs": old_proofs,
        "excluded_prior_claims": proposal["excluded_prior_claims"],
        "slots": slots,
        "configs": {a: chz.asdict(c) for a, c in configs.items()},
        "checkpoints": original_identity["checkpoints"],
        "qualified": qualified,
        "execution_sources": sources,
        "versions": versions,
        "maximum_new_rollouts": 138,
        "retry_permitted": False,
        "capacity_policy": original.CAPACITY_POLICY,
        "guard": "exact_raw_parse_error_v1",
        "automatic_training": False,
        "original_results_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "identity.json").exists():
            if read_json(output / "identity.json") != identity:
                raise ValueError("Continuation identity changed")
        else:
            exclusive_json(output / "identity.json", identity)
        pairs = {
            a: [(str(s["task_name"]), int(str(s["sample_index"]))) for s in slots if s["arm"] == a]
            for a in original.ARMS
        }
        for arm in original.ARMS:
            arm_identity = {
                "phase": original.pinned_file(output / "identity.json"),
                "arm": arm,
                "checkpoint": mapping(identity["checkpoints"])[arm],
            }
            p = output / arm / "identity.json"
            if p.exists():
                if read_json(p) != arm_identity:
                    raise ValueError("Arm identity changed")
            else:
                exclusive_json(p, arm_identity)
        pending = check_prior(output, pairs)
        preflight = {
            "identity_sha256": digest(identity),
            "pending": {a: len(p) for a, p in pending.items()},
            "maximum_new_rollouts": 138,
            "excluded_prior_claims": 22,
            "dispatch": config.dispatch,
            "model_requests": 0,
            "sandbox_requests": 0,
        }
        original.write_json(output / "preflight.json", preflight)
        print(json.dumps(preflight), flush=True)
        if not config.dispatch:
            return
        validate_review(config.review_receipt, identity)
        if not any(pending.values()):
            return
        original.write_json(output / "capacity_startup.json", capacity())
        original.load_env_file(Path(config.env_file))
        cache = output / "contree_images.json"
        if not cache.exists():
            shutil.copyfile(old / "contree_images.json", cache)
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
        for arm in original.ARMS:
            chosen = pending[arm]
            if not chosen:
                continue
            original.write_json(output / f"capacity_{arm}.json", capacity())
            original.check_proofs([cast(dict[str, str], p) for p in sources])
            original.check_proofs(old_proofs)
            select_slots(old, proposal, audit, original_identity)
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
                original.check_proofs([cast(dict[str, str], p) for p in sources])
                original.check_proofs(old_proofs)
                select_slots(old, proposal, audit, original_identity)
                task = by_name[pair[0]]
                if original.current_task_digest(task) != mapping(manifest["task_hashes"])[pair[0]]:
                    raise ValueError("Task changed")
                result = await original.evaluate_task(
                    task, policy, renderer, factory, cfg, folder, result_lock, tokenizer, pair[1]
                )
                parse_stop_guard(folder, pair, result)
                return result

            results = await original.dispatch_slots(folder, chosen, operation, 4)
            errors = [f"{type(r).__name__}: {r}" for r in results if isinstance(r, BaseException)]
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
                "original_results_modified": False,
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
            and read_json(output / "identity.json").get("phase_kind")
            == "donor_eval_never_claimed_v1"
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
