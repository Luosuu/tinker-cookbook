"""Credential-free Slurm worker for the expansion coordinator."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from .expansion import atomic_json
from .models import KokkosInstance


async def run(input_path: Path, output: Path) -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME") not in {
        "rivanna",
        "shen",
    }:
        raise RuntimeError("Worker requires a Rivanna Slurm allocation")
    request = json.loads(input_path.read_text())
    from .expansion_batch import source_digest

    if request.get("source_sha256") != source_digest():
        raise ValueError("Worker source differs from coordinator")
    instance = KokkosInstance.from_dict(request["instance"])
    result = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "cluster": os.environ["SLURM_CLUSTER_NAME"],
        "request_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    try:
        from .qualification import execution_policy

        policy = execution_policy(instance)
        result["execution_policy"] = policy
        if os.environ.get("KOKKOS_RIVANNA_BASE_SIF") != request["base_sif"]:
            raise ValueError("Base image configuration mismatch")
        with Path(request["base_sif"]).open("rb") as base_image:
            result["base_sif_sha256"] = hashlib.file_digest(base_image, "sha256").hexdigest()
        if request["action"] == "construction":
            from .modal_validate import validate_instance_in_sandbox
            from .rivanna_validate import rivanna_validation_sandbox_factory

            report = await validate_instance_in_sandbox(
                instance,
                command_timeout=policy["command_seconds"],
                sandbox_timeout=policy["sandbox_seconds"],
                flaky_repetitions=3,
                sandbox_factory=rivanna_validation_sandbox_factory,
            )
            result["validation"] = report.to_dict()
        elif request["action"] in {"preflight", "target_discovery"}:
            from .preflight import inventory

            result["preflight"] = await inventory(
                instance, discovery_only=request["action"] == "target_discovery"
            )
        elif request["action"] == "acceptance":
            from .expansion_batch import build_export, exported_trials, tree_digest
            from .export_harbor import export_instance

            task = export_instance(instance, output.parent / "export")
            sif = await build_export(
                task,
                output.parent,
                timeout=policy["image_seconds"],
                parallelism=policy["build_parallelism"],
            )
            result["export_sha256"] = tree_digest(task)
            with sif.open("rb") as handle:
                result["sif_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
            result["trials"] = await exported_trials(instance, task, output.parent, sif)
        else:
            raise ValueError("Unknown worker action")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    atomic_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output))


if __name__ == "__main__":
    main()
