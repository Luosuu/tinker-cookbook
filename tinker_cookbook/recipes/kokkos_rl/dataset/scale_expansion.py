"""Sharded, dollar-capped training-task expansion over mined candidates.

The gates are those of ``expansion_batch``: exclusion against the frozen benchmark and prior
admissions, upstream tests on a supported CPU profile, preflight inventory, model annotation
validated by Rivanna execution, a public-source statement with an independent audit, a fresh
export and the five-trial independent acceptance. This driver adds a preflight before any
model call (its target and gtest inventory go into the candidate metadata the annotator
sees), deterministic sharding so several coordinators can run in parallel against separate
Rivanna roots, and a shared dollar cap over all model requests.

Usage, one process per shard::

    python -m tinker_cookbook.recipes.kokkos_rl.dataset.scale_expansion \\
        --candidates core/candidates.jsonl --candidates kernels/candidates.jsonl \\
        --frozen-instances data/kokkos/SWE-kokkos-bench-v2.5/instances.jsonl \\
        --prior-accepted prior.jsonl --output-dir CAMPAIGN --shard 0 --shards 8 \\
        --rivanna-root /scratch/USER/CAMPAIGN --rivanna-base-sif /scratch/USER/base.sif \\
        --ledger CAMPAIGN/spend.json --max-usd 100
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..spend import BudgetStop, SpendLedger, capped_chat_client
from .auto_annotate import CSTDINT_COMPAT_FLAG
from .expansion import atomic_json, digest, exclusion, rows, write_rows
from .expansion_batch import (
    check_acceptance,
    construction_failure,
    failure_class,
    source_digest,
    statement_risks,
    tree_digest,
)
from .models import KokkosInstance

# Annotation and statement prompts carry the candidate patch, CMake context and evidence;
# reserve as if each were this large and uncached.
MAX_PROMPT_TOKENS = 60_000


def shard_of(instance_id: str, shards: int) -> int:
    return int(hashlib.sha256(instance_id.encode()).hexdigest(), 16) % shards


def shard_candidates(paths: list[Path], shard: int, shards: int) -> list[dict]:
    seen: set[str] = set()
    selected = []
    for path in paths:
        for row in rows(path):
            ident = row["instance_id"]
            if ident in seen:
                continue
            seen.add(ident)
            if shard_of(ident, shards) == shard:
                selected.append(row)
    return selected


def profile_reason(instance: KokkosInstance) -> str | None:
    if instance.metadata.get("test_origin") != "upstream":
        return "Model-generated/unknown test origin requires a separate validation lane"
    if instance.repo not in {"kokkos/kokkos", "kokkos/kokkos-kernels"}:
        return "Unsupported Rivanna environment profile"
    if instance.metadata.get("accelerator") != "cpu":
        return "Unsupported Rivanna environment profile"
    if instance.repo == "kokkos/kokkos-kernels" and not str(
        instance.metadata.get("kokkos_dependency_commit", "")
    ):
        return "Historical kernels environment needs a pinned dependency"
    return None


def needs_cstdint(result: dict) -> bool:
    found = result.get("preflight") or {}
    if found.get("stage") != "build" or not found.get("records"):
        return False
    last = found["records"][-1]
    return bool(
        re.search(r"[‘']uint32_t[’'] has not been declared", last["stdout"] + last["stderr"])
    )


def with_cstdint(instance: KokkosInstance) -> KokkosInstance:
    return replace(
        instance,
        configure_command=instance.configure_command + " " + CSTDINT_COMPAT_FLAG,
        metadata={**instance.metadata, "toolchain_compat": "include-cstdint"},
    )


async def process(instance: KokkosInstance, *, directory: Path, bridge, completer, args) -> dict:
    from .auto_annotate import AutoAnnotationResult, annotate_and_validate_instance
    from .export_harbor import export_instance
    from .mine import GitHubAPI
    from .preflight import require_baseline_selection
    from .revise_instructions import revise_and_audit
    from .rivanna_validate import rivanna_validation_sandbox_factory

    receipt: dict = {"instance_id": instance.instance_id, "stage": "preflight"}
    atomic_json(directory / "progress.json", receipt)
    preflight_path = directory / "preflight.json"
    result = (
        json.loads(preflight_path.read_text())
        if preflight_path.exists()
        else await bridge.execute(instance, "preflight")
    )
    atomic_json(preflight_path, result)
    if needs_cstdint(result):
        instance = with_cstdint(instance)
        compat_path = directory / "preflight-cstdint.json"
        result = (
            json.loads(compat_path.read_text())
            if compat_path.exists()
            else await bridge.execute(instance, "preflight")
        )
        atomic_json(compat_path, result)
    found = result.get("preflight") or {}
    if result.get("error") or not found.get("passed") or not found.get("test_names"):
        raise ValueError(
            f"preflight {found.get('stage', 'environment_error')}: {result.get('error', '')}"
        )
    metadata = dict(instance.metadata)
    metadata["runtime_test_inventory"] = {k: found[k] for k in ("target", "binary", "test_names")}
    metadata["preflight_evidence_sha256"] = digest(result)
    instance = replace(instance, metadata=metadata)

    receipt["stage"] = "construction"
    atomic_json(directory / "progress.json", receipt)
    construction_path = directory / "construction.json"
    if construction_path.exists():
        annotated = AutoAnnotationResult.from_dict(json.loads(construction_path.read_text()))
    else:
        annotated = await annotate_and_validate_instance(
            instance,
            completer=completer,
            max_attempts=args.max_attempts,
            sandbox_timeout=1200,
            command_timeout=300,
            flaky_repetitions=3,
            sandbox_factory=rivanna_validation_sandbox_factory,
            validation_runner=bridge.validate,
        )
        atomic_json(construction_path, annotated.to_dict())
    if not annotated.passed or annotated.validated_instance is None:
        last = annotated.attempts[-1] if annotated.attempts else None
        raise ValueError(
            (last.error or construction_failure(last.validation or {}))
            if last
            else "Missing construction evidence"
        )
    instance = annotated.validated_instance
    if not instance.p2p_commands:
        raise ValueError("Missing P2P regression coverage")
    require_baseline_selection(instance)

    receipt["stage"] = "instruction_audit"
    atomic_json(directory / "progress.json", receipt)
    instance, statement_passed = await revise_and_audit(
        instance,
        completer=completer,
        api=GitHubAPI(os.environ.get("GITHUB_TOKEN")),
        reports_dir=directory / "statement",
        max_revisions=args.max_statement_revisions,
        inference_config={
            "model": args.model,
            "base_url": args.base_url,
            "provider_parameters": args.provider_parameters,
            "max_tokens": args.max_tokens,
            "source_sha256": source_digest(),
        },
    )
    if not statement_passed:
        raise ValueError("Uncertain or unsupported statement after bounded revision")
    risks = statement_risks(instance)
    if risks:
        raise ValueError(f"statement risks: {risks}")

    receipt["stage"] = "independent_acceptance"
    atomic_json(directory / "progress.json", receipt)
    task = export_instance(instance, directory / "export")
    export_sha = tree_digest(task)
    evidence = await bridge.execute(instance, "acceptance")
    if evidence.get("error"):
        raise RuntimeError(str(evidence["error"]))
    check_acceptance(evidence, export_sha)
    atomic_json(directory / "validated.json", instance.to_dict())
    return {"export_sha256": export_sha, "trials": evidence["trials"], "validated": instance}


async def run(args: argparse.Namespace) -> None:
    from ..chat_inference import ChatAnnotationCompleter
    from .auto_annotate import _load_env_file
    from .rivanna_bridge import RivannaBridge

    _load_env_file(args.env_file)
    root = args.output_dir.resolve() / f"shard-{args.shard:02d}"
    root.mkdir(parents=True, exist_ok=True)
    frozen = rows(args.frozen_instances)
    prior = [row for path in args.prior_accepted for row in rows(path)]
    candidates = shard_candidates(args.candidates, args.shard, args.shards)
    config = {
        "version": 1,
        "source_sha256": source_digest(),
        "candidates_sha256": digest(candidates),
        "frozen_sha256": digest(frozen),
        "prior_sha256": digest(prior),
        "shard": [args.shard, args.shards],
        "model": args.model,
        "base_url": args.base_url,
        "provider_parameters": args.provider_parameters,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "max_statement_revisions": args.max_statement_revisions,
        "rivanna_root": f"{args.rivanna_root}/shard-{args.shard:02d}",
        "rivanna_base_sif": args.rivanna_base_sif,
        "rivanna_partition": args.rivanna_partition,
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Shard inputs/configuration changed; use a fresh campaign directory")
    atomic_json(config_path, config)
    bridge = RivannaBridge(
        root / "remote",
        host=args.rivanna_host,
        remote_root=config["rivanna_root"],
        source_sha256=config["source_sha256"],
        base_sif=args.rivanna_base_sif,
        partition=args.rivanna_partition,
    )
    ledger = SpendLedger(args.ledger, args.max_usd)
    completer = ChatAnnotationCompleter(
        model=args.model,
        max_tokens=args.max_tokens,
        thinking_effort=0.7,
        reports_dir=root / "api_calls",
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=args.provider_parameters,
    )
    completer.client = completer.client.with_options(max_retries=0)
    capped_chat_client(
        completer.client, ledger, label=f"shard-{args.shard}", max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    accepted: list[dict] = []
    records: list[dict] = []
    start = time.monotonic()

    def report(status: str) -> None:
        write_rows(root / "accepted_training.jsonl", accepted)
        atomic_json(
            root / "report.json",
            {
                "status": status,
                "candidates": len(candidates),
                "processed": len(records),
                "accepted": len(accepted),
                "failure_classes": dict(
                    Counter(r.get("failure_class", "accepted") for r in records)
                ),
                "stages_reached": dict(Counter(r.get("stage") for r in records)),
                "campaign_spend_usd": round(ledger.spent(), 4),
                "elapsed_seconds": round(time.monotonic() - start, 1),
            },
        )

    try:
        for candidate in candidates:
            instance = KokkosInstance.from_dict(candidate)
            directory = root / instance.instance_id
            directory.mkdir(exist_ok=True)
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["config_sha256"] != digest(config):
                    raise ValueError(f"Stale receipt for {instance.instance_id}")
                if receipt["status"] == "accepted":
                    accepted.append(json.loads((directory / "validated.json").read_text()))
                records.append(receipt)
                continue
            receipt = {
                "instance_id": instance.instance_id,
                "input_sha256": digest(candidate),
                "config_sha256": digest(config),
                "status": "quarantined",
                "stage": "exclusion",
            }
            began = time.monotonic()
            try:
                reason = exclusion(candidate, frozen, prior + accepted) or profile_reason(instance)
                if reason:
                    raise ValueError(reason)
                outcome = await process(
                    instance, directory=directory, bridge=bridge, completer=completer, args=args
                )
                receipt.update(
                    status="accepted",
                    stage="accepted",
                    export_sha256=outcome["export_sha256"],
                    trials=outcome["trials"],
                )
                accepted.append(outcome["validated"].to_dict())
            except BudgetStop:
                raise
            except Exception as error:
                progress = directory / "progress.json"
                if progress.exists():
                    receipt["stage"] = json.loads(progress.read_text()).get(
                        "stage", receipt["stage"]
                    )
                receipt["failure_class"] = failure_class(str(error))
                text = str(error)[-2000:]
                for key, secret in os.environ.items():
                    if any(word in key for word in ("KEY", "TOKEN", "SECRET")) and len(secret) > 8:
                        text = text.replace(secret, "[REDACTED]")
                receipt["reason"] = text
            receipt["elapsed_seconds"] = round(time.monotonic() - began, 2)
            receipt["evidence_hashes"] = {
                str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*")
                if p.is_file() and p != receipt_path and p.suffix != ".sif"
            }
            atomic_json(receipt_path, receipt)
            records.append(receipt)
            report("running")
            print(instance.instance_id, receipt["status"], receipt["stage"], flush=True)
        report("complete")
    except BudgetStop as stop:
        report(f"budget_stop: {stop}")
        print("Budget stop:", stop, flush=True)
    finally:
        await completer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--frozen-instances", type=Path, required=True)
    parser.add_argument("--prior-accepted", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--rivanna-host", default="login.hpc.virginia.edu")
    parser.add_argument("--rivanna-root", required=True)
    parser.add_argument("--rivanna-base-sif", required=True)
    parser.add_argument("--rivanna-partition", default="bii-largemem")
    parser.add_argument(
        "--base-url", default="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    )
    parser.add_argument("--model", default="thinkingmachines/Inkling-Small:peft:262144")
    parser.add_argument("--api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--provider-parameters",
        type=json.loads,
        default={"reasoning_effort": 0.7, "separate_reasoning": True},
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-statement-revisions", type=int, default=2)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be in [0, shards)")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
