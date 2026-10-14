"""Local model / credential-free Rivanna validation bridge with bounded Slurm jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import time
from pathlib import Path

from .expansion import atomic_json, digest
from .models import KokkosInstance
from .validate import CommandResult, ValidationReport

# Parallel coordinators share one SSH connection; the login node drops bursts of new ones.
SSH_OPTIONS = (
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPath=~/.ssh/cm-%C",
    "-o",
    "ControlPersist=600",
)
MAX_QUEUE_SECONDS = 4 * 24 * 3600
# Refusals during the handshake, before any remote command runs, so a retry cannot repeat one.
HANDSHAKE_REFUSAL = re.compile(
    r"kex_exchange_identification|Connection (?:closed|reset) by \S+ port 22"
    r"|Connection timed out during banner exchange"
)


async def _command(*args: str, timeout: int = 60) -> str:
    if args[0] in {"ssh", "scp"}:
        args = (args[0], *SSH_OPTIONS, *args[1:])
    for attempt in range(6):
        try:
            return await _command_once(*args, timeout=timeout)
        except RuntimeError as error:
            if attempt == 5 or not HANDSHAKE_REFUSAL.search(str(error)):
                raise
            await asyncio.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


async def _command_once(*args: str, timeout: int) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise TimeoutError(f"{args[0]} timed out") from None
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed: " + stderr.decode(errors="replace")[-2000:])
    return stdout.decode()


class RivannaBridge:
    def __init__(
        self,
        local_root: Path,
        *,
        host: str,
        remote_root: str,
        source_sha256: str,
        base_sif: str = "",
        partition: str = "standard",
        cuda_base_sifs: dict[str, str] | None = None,
    ):
        if not re.fullmatch(r"[A-Za-z0-9.@_-]+", host) or not re.fullmatch(
            r"/[A-Za-z0-9/_-]+", remote_root
        ):
            raise ValueError("Unsafe bridge location")
        self.local_root, self.host, self.remote_root = local_root, host, remote_root
        if base_sif and not re.fullmatch(r"/[A-Za-z0-9/_.-]+", base_sif):
            raise ValueError("Unsafe base image path")
        if partition not in {"standard", "bii-largemem"}:
            raise ValueError("Unsupported Rivanna CPU partition")
        self.cuda_base_sifs = dict(cuda_base_sifs or {})
        if any(not re.fullmatch(r"/[A-Za-z0-9/_.-]+", p) for p in self.cuda_base_sifs.values()):
            raise ValueError("Unsafe base image path")
        self.source_sha256 = source_sha256
        self.base_sif, self.partition = base_sif, partition

    def allocation(self, instance: KokkosInstance) -> dict:
        """CPU jobs on the configured partition; CUDA jobs on one Ampere GPU with an era image."""
        if instance.metadata.get("toolchain") != "cuda":
            return {
                "base_sif": self.base_sif,
                "partition": self.partition,
                "gres": "",
                "cpus": 4,
                "memory": "16G",
                "time": "00:30:00",
                "poll_seconds": 3600,
            }
        base = self.cuda_base_sifs.get(instance.era, "")
        if not base:
            raise ValueError(f"No CUDA base image for era {instance.era}")
        # AMPERE80/VOLTA70 builds run on any Ampere GPU; Kokkos refuses them on Blackwell.
        return {
            "base_sif": base,
            "partition": "gpu",
            "gres": "gpu:1",
            "constraint": "a6000|a40|a100",
            "cpus": 8,
            "memory": "64G",
            "time": "03:00:00",
            "poll_seconds": 6 * 3600,
        }

    async def execute(self, instance: KokkosInstance, action: str) -> dict:
        resources = self.allocation(instance)
        if not resources["base_sif"]:
            raise ValueError("An explicit prepared Rivanna base SIF is required")
        request = {
            "instance": instance.to_dict(),
            "action": action,
            "source_sha256": self.source_sha256,
            "base_sif": resources["base_sif"],
            "partition": resources["partition"],
        }
        identity = digest(request)
        local = self.local_root / identity
        local.mkdir(parents=True, exist_ok=True)
        remote = self.remote_root + "/jobs/" + identity
        request_path = local / "request.json"
        atomic_json(request_path, request)
        output = local / "result.json"
        job_path = local / "job.json"
        if output.exists() and (local / "evidence.tar.gz").exists():
            result = json.loads(output.read_text())
            if (
                result.get("request_sha256")
                != hashlib.sha256(request_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Stale remote result")
            return result
        if not job_path.exists():
            gres_line = (
                f"#SBATCH --gres={resources['gres']}\n#SBATCH --constraint={resources['constraint']}\n"
                if resources["gres"]
                else ""
            )
            script = f"""#!/bin/bash
#SBATCH --partition={resources["partition"]}
#SBATCH --account=lia-lab-members
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources["cpus"]}
#SBATCH --mem={resources["memory"]}
#SBATCH --time={resources["time"]}
{gres_line}#SBATCH --job-name=kokkos-expand-{action}
#SBATCH --output={remote}/slurm.log
source /etc/profile
set -euo pipefail
module load apptainer/1.4.5
cd {remote}
export PYTHONPATH={self.remote_root}/code
export PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR=/scratch/fad3ew/openhands-smoke/cache
export APPTAINER_TMPDIR=$(mktemp -d /tmp/kokkos-expansion-worker.XXXXXX)
trap 'rm -rf -- "$APPTAINER_TMPDIR"' EXIT
export KOKKOS_RIVANNA_BASE_SIF={resources["base_sif"]}
export CMAKE_BUILD_PARALLEL_LEVEL={resources["cpus"] if resources["gres"] else 2}
export OMP_NUM_THREADS=2
/scratch/fad3ew/openhands-smoke/.venv/bin/python -m tinker_cookbook.recipes.kokkos_rl.dataset.rivanna_worker --input request.json --output result.json
"""
            (local / "run.sbatch").write_text(script)
            await _command("ssh", "-o", "BatchMode=yes", self.host, "mkdir -p " + remote)
            await _command(
                "scp", str(request_path), str(local / "run.sbatch"), self.host + ":" + remote + "/"
            )
            job = (
                (
                    await _command(
                        "ssh",
                        self.host,
                        "bash -lc "
                        + shlex.quote("cd " + remote + " && sbatch --parsable run.sbatch"),
                    )
                )
                .strip()
                .split(";")[0]
            )
            if not job.isdigit():
                raise RuntimeError("Invalid sbatch receipt")
            atomic_json(job_path, {"job_id": job, "remote": remote, "request_sha256": identity})
            print("Rivanna submitted", job, instance.instance_id, action, flush=True)
        job = json.loads(job_path.read_text())["job_id"]
        # Only running time counts against the job's limit: after a maintenance window the GPU
        # queue can hold jobs for more than a day, and a queued job is not evidence about a task.
        queued = time.monotonic()
        running = None
        while True:
            now = time.monotonic()
            if running is None and now - queued > MAX_QUEUE_SECONDS:
                break
            if running is not None and now - running > resources["poll_seconds"]:
                break
            state = (
                (
                    await _command(
                        "ssh",
                        self.host,
                        "bash -lc " + shlex.quote(f"sacct -j {job} -nP -X -o State"),
                    )
                )
                .strip()
                .splitlines()
            )
            status = state[0].split("|")[0] if state else ""
            if running is None and status in {"RUNNING", "COMPLETING", "COMPLETED"}:
                running = now
            if status == "COMPLETED":
                await _command("scp", self.host + ":" + remote + "/result.json", str(output))
                # Fetch only evidence, not images or environments.
                await _command(
                    "ssh",
                    self.host,
                    "cd "
                    + remote
                    + " && tar -czf "
                    + remote
                    + ".evidence.tar.gz --exclude=task.sif --exclude=tmp --exclude=evidence.tar.gz .",
                    timeout=120,
                )
                await _command(
                    "scp",
                    self.host + ":" + remote + ".evidence.tar.gz",
                    str(local / "evidence.tar.gz"),
                    timeout=120,
                )
                result = json.loads(output.read_text())
                if (
                    result.get("request_sha256")
                    != hashlib.sha256(request_path.read_bytes()).hexdigest()
                ):
                    raise ValueError("Remote request digest mismatch")
                return result
            if status and status not in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "REQUEUED",
            }:
                await _command(
                    "scp", self.host + ":" + remote + "/slurm.log", str(local / "slurm.log")
                )
                raise RuntimeError("Rivanna job " + job + ": " + status)
            await asyncio.sleep(15 if running is not None else 60)
        raise TimeoutError("Rivanna job remains pending/running; retained job receipt for resume")

    async def validate(self, instance: KokkosInstance, **kwargs: object) -> ValidationReport:
        result = await self.execute(instance, "construction")
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        report = result["validation"]
        return ValidationReport(
            instance_id=report["instance_id"],
            passed=report["passed"],
            results=[CommandResult(**row) for row in report["results"]],
            error=report.get("error", ""),
        )
