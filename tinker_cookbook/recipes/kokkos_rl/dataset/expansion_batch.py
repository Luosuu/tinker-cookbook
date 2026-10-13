"""Rivanna-only automatic construction, audit, export and acceptance of mined candidates.

Conservative first lane: upstream tests, CPU core tasks, unchanged clear statements.
Failures and missing evidence are quarantined, never emitted as training examples.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import time
from collections import Counter
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.chat_inference import ChatAnnotationCompleter

from .expansion import atomic_json, digest, exclusion, rows, write_rows
from .models import KokkosInstance


def source_digest() -> str:
    root = Path(__file__).parent
    paths = [p for p in root.glob("*.py") if not p.name.endswith("_test.py")] + [
        root.parent / "chat_inference.py",
        root.parents[2] / "sandbox/apptainer_sandbox.py",
        root.parents[1] / "harbor_rl/harbor_tools.py",
    ]
    return digest(
        {
            str(p.relative_to(root.parents[3])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        }
    )


def failure_class(text: str) -> str:
    text = text.lower()
    if "unstable" in text or "flaky" in text:
        return "unstable"
    if any(s in text for s in ("timeout", "timed out", "exit 124")):
        return "timeout"
    if any(
        s in text
        for s in (
            "sandbox",
            "apptainer",
            "checkout",
            "no space",
            "memory",
            "connection",
            "not found",
            "unauthorized",
            "authentication",
            "http",
            "rivanna job",
            "environment profile",
            "environment_error",
        )
    ):
        return "environment_error"
    return "invalid_or_unresolved_task"


def construction_failure(report: dict) -> str:
    """Detect inconsistent repeated outcomes within the same repository state."""
    phase = "base"
    outcomes: dict[tuple[str, str, str], bool] = {}
    for row in report.get("results", []):
        command = row["command"]
        if command.startswith("git apply") and "kokkos-test.patch" in command:
            phase = "test-only"
        if command.startswith("git apply") and "kokkos-code.patch" in command:
            phase = "gold"
        code = row["exit_code"]
        if code < 0 or code in {124, 126, 127, 137, 143}:
            diagnostics = " ".join(str(row.get(key, "")) for key in ("stdout", "stderr"))
            timed_out = code == 124 or "timed out" in diagnostics.lower()
            return (
                ("timeout" if timed_out else "environment_error") + ": " + report.get("error", "")
            )
        key = (phase, command, row["expected_exit"])
        passed = code == 0
        if key in outcomes and outcomes[key] != passed:
            return "unstable: repeated outcomes differ in " + phase
        outcomes[key] = passed
    return report.get("error", "Construction did not pass")


def statement_risks(instance: KokkosInstance) -> list[str]:
    statement = instance.problem_statement
    reasons = []
    if len(statement.strip()) < 80:
        reasons.append("insufficient-statement")
    if re.search(
        r"https?://\S*(?:pull/|commit/)|gold\.patch|test\.patch|diff --git|@@ -", statement
    ):
        reasons.append("reference-history-or-patch-leak")
    private_lines = [
        line[1:].strip()
        for line in instance.code_patch.splitlines()
        if line.startswith("+") and not line.startswith("+++") and len(line.strip()) > 40
    ]
    if any(line in statement for line in private_lines):
        reasons.append("reference-implementation-overlap")
    from .patching import split_unified_diff

    if any(path in statement for path, _ in split_unified_diff(instance.test_patch)):
        reasons.append("hidden-test-path")
    if any(
        command in statement
        for command in instance.f2p_commands + instance.p2p_commands
        if len(command) > 20
    ):
        reasons.append("hidden-test-command")
    return reasons


def check_acceptance(evidence: dict, export_digest: str) -> None:
    if evidence.get("error") or evidence.get("export_sha256") != export_digest:
        raise ValueError("Missing, failed or mismatched export evidence")
    trials = evidence.get("trials", [])
    modes = ["nop", "oracle", "nop", "oracle", "tampered-nop"]
    if len(trials) != len(modes) or not evidence.get("sif_sha256") or not evidence.get("job_id"):
        raise ValueError("Incomplete independent acceptance evidence")
    for trial, mode in zip(trials, modes, strict=True):
        if (
            trial.get("mode") != mode
            or trial.get("reward") != int(mode == "oracle")
            or trial.get("isolation_probe") is not True
            or not trial.get("sha256")
        ):
            raise ValueError("Unstable or incomplete independent acceptance trial")


def tree_digest(directory: Path) -> str:
    return digest(
        {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "sif.path"
        }
    )


async def build_export(
    task: Path, output: Path, *, timeout: int = 900, parallelism: int = 2
) -> Path:
    """Translate the exported Dockerfile faithfully, adding only the sandbox server."""
    if not 1 <= timeout <= 2400:
        raise ValueError("Image command budget exceeds 2400 seconds")
    commands = []
    base = None
    for line in (task / "environment/Dockerfile").read_text().replace("\\\n", " ").splitlines():
        if not line.strip():
            continue
        op, value = line.split(" ", 1)
        if op == "FROM":
            base = value
        elif op == "ENV":
            commands.append("export " + value)
        elif op == "RUN":
            commands.append(value)
        elif op != "WORKDIR":
            raise ValueError("Unsupported Dockerfile directive: " + op)
    if not base:
        raise ValueError("Missing export base image")
    definition = output / "task.def"
    definition.write_text(
        "Bootstrap: docker\nFrom: "
        + base
        + "\n\n%post\n    set -eu\n    "
        + "\n    ".join(commands).replace(" --parallel", f" --parallel {parallelism}")
        + "\n    apt-get update && apt-get install -y python3-venv tini tmux\n"
        "    python3 -m venv /opt/uv-bootstrap\n"
        "    /opt/uv-bootstrap/bin/python -m pip install uv==0.7.2\n"
        "    export UV_PYTHON_INSTALL_DIR=/opt/python\n"
        "    /opt/uv-bootstrap/bin/uv python install 3.12\n"
        "    /opt/uv-bootstrap/bin/uv venv --python 3.12 /opt/openhands\n"
        "    /opt/uv-bootstrap/bin/uv pip install --python /opt/openhands/bin/python openhands-agent-server==1.45.0 openhands-sdk==1.45.0 openhands-tools==1.45.0\n"
        "    mkdir -p /root /logs/verifier\n\n%environment\n"
        "    export PATH=/opt/openhands/bin:$PATH\n    export OH_ENABLE_VSCODE=false\n\n%runscript\n"
        '    cd /workspace/repo\n    exec /usr/bin/tini -- /opt/openhands/bin/python -m openhands.agent_server "$@"\n'
    )
    sif = output / "task.sif"
    with (output / "image-build.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            "apptainer", "build", "--fakeroot", str(sif), str(definition), stdout=log, stderr=log
        )
        try:
            code = await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Export image build exceeded {timeout}s") from None
    if code:
        raise RuntimeError("Apptainer exported image build failed; inspect image-build.log")
    return sif


def validate_trial_checks(instance: KokkosInstance, mode: str, log: dict) -> str:
    """Require a causal negative result, never count missing checks as evidence."""
    details = log.get("diagnostics", {})
    checks = details.get("checks", [])
    expected = ["setup", "build"] + [f"f2p-{i}" for i in range(len(instance.f2p_commands))] + [f"p2p-{i}" for i in range(len(instance.p2p_commands))]
    if (not details.get("complete") or [c.get("id") for c in checks] != expected
            or any(c.get("status") not in {"passed", "failed", "blocked"} for c in checks)):
        raise RuntimeError("Incomplete or infrastructure-failed exported checks")
    if mode == "oracle":
        if not all(c["status"] == "passed" for c in checks):
            raise ValueError("Oracle requires every check to pass")
        return "all_checks_passed"
    failures = [c for c in checks if c["status"] == "failed"]
    if not failures or any(c.get("exit_code") in {None, 0, 124, 126, 127, 137, 143} for c in failures):
        raise ValueError("Negative trial has no ordinary failure evidence")
    failed_ids = {c["id"] for c in failures}
    for i, check in enumerate(checks):
        if check["status"] == "blocked" and (
            check.get("blocked_by") not in failed_ids
            or check.get("blocked_by") not in {c["id"] for c in checks[:i]}
        ):
            raise ValueError("Blocked check lacks a preceding causal failure")
    if mode == "tampered-nop":
        if (failed_ids != {"setup"} or any(c["status"] != "blocked" or c.get("blocked_by") != "setup" for c in checks[1:])
                or "candidate patch changes protected test/build/CI files" not in log.get("stdout", "") + log.get("stderr", "")):
            raise ValueError("Tamper probe did not explicitly reject protected edits")
        return "protected_edit_rejected"
    if mode != "nop" or checks[0]["status"] != "passed":
        raise ValueError("NOP setup must pass")
    if instance.metadata.get("f2p_stage") == "build":
        if failed_ids != {"build"} or any(c["status"] != "blocked" or c.get("blocked_by") != "build" for c in checks[2:]):
            raise ValueError("Compile F2P requires an observed build failure")
        if "error:" not in log.get("stdout", "") + log.get("stderr", ""):
            raise ValueError("Compile F2P lacks compiler error evidence")
        return "compile_regression_observed"
    if checks[1]["status"] != "passed" or any(c["status"] == "blocked" for c in checks) or any(not c["id"].startswith("f2p-") for c in failures):
        raise ValueError("Runtime NOP requires executed F2P failure and passing P2P")
    return "runtime_regression_observed"


async def exported_trials(
    instance: KokkosInstance, task: Path, root: Path, sif: Path
) -> list[dict]:
    from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
    from tinker_cookbook.sandbox.apptainer_sandbox import ApptainerSandbox

    from .patching import split_unified_diff

    from .qualification import execution_policy
    policy = execution_policy(instance)
    trials = []
    for mode in ("nop", "oracle", "nop", "oracle", "tampered-nop"):
        sandbox = await ApptainerSandbox.create(
            sif,
            policy["sandbox_seconds"],
            default_workdir="/workspace/repo",
            enable_gpu=instance.metadata.get("toolchain") == "cuda",
            allow_network=False,
            command_env={
                "CMAKE_BUILD_PARALLEL_LEVEL": str(policy["build_parallelism"]),
                "OMP_NUM_THREADS": "2",
            },
        )
        try:
            # Before uploading trusted grading materials, only the base commit may be visible.
            probe = await sandbox.run_command(
                "test ! -e /tests && test ! -e /solution && test ! -e /tmp/kokkos-code.patch && "
                f'test "$(git rev-list --all)" = {shlex.quote(instance.base_commit)} && '
                f"! git cat-file -e {shlex.quote(instance.merge_commit)}",
                timeout=30,
            )
            if probe.exit_code:
                raise RuntimeError("Hidden material / reference history isolation probe failed")
            if mode == "oracle":
                result = await sandbox.write_file(
                    "/tmp/candidate.patch", (task / "solution/gold.patch").read_bytes()
                )
                if result.exit_code:
                    raise RuntimeError("Oracle patch upload failed")
                result = await sandbox.run_command("git apply /tmp/candidate.patch")
                if result.exit_code:
                    raise RuntimeError("Oracle patch application failed")
            elif mode == "tampered-nop":
                protected = split_unified_diff(instance.test_patch)[0][0]
                result = await sandbox.write_file(
                    "/workspace/repo/" + protected, "// tampered test\n"
                )
                if result.exit_code:
                    raise RuntimeError("Anti-bypass probe setup failed")
            path = root / f"trial-{len(trials)}-{mode}.json"
            reward, _ = await HarborReward(
                tests_dir=task / "tests",
                sandbox=sandbox,
                grader_timeout=policy["grader_seconds"],
                raise_on_grading_error=True,
                grading_log_path=path,
            )([])
            failure_basis = validate_trial_checks(instance, mode, json.loads(path.read_text()))
            if reward != float(mode == "oracle"):
                raise ValueError("Unstable or invalid export: " + mode)
            trials.append(
                {
                    "mode": mode,
                    "reward": reward,
                    "evidence": path.name,
                    "failure_basis": failure_basis,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "isolation_probe": True,
                }
            )
        finally:
            await sandbox.cleanup()
    return trials


async def run(args: argparse.Namespace) -> None:
    from .auto_annotate import AutoAnnotationResult, _load_env_file, annotate_and_validate_instance
    from .export_harbor import export_instance
    from .rivanna_bridge import RivannaBridge
    from .rivanna_validate import rivanna_validation_sandbox_factory

    bridge = RivannaBridge(
        args.output_dir.resolve() / "remote",
        host=args.rivanna_host,
        remote_root=args.rivanna_root,
        source_sha256=source_digest(),
        base_sif=args.rivanna_base_sif,
        partition=args.rivanna_partition,
    )
    _load_env_file(args.env_file)
    frozen = rows(args.frozen_instances)
    candidates = rows(args.candidates)[: args.limit]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "source_sha256": source_digest(),
        "candidates_sha256": digest(candidates),
        "frozen_sha256": digest(frozen),
        "model": args.model,
        "base_url": args.base_url,
        "provider_parameters": args.provider_parameters,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "max_statement_revisions": args.max_statement_revisions,
        "repetitions": 3,
        "backend": "rivanna",
        "rivanna_host": args.rivanna_host,
        "rivanna_root": args.rivanna_root,
        "rivanna_base_sif": args.rivanna_base_sif,
        "rivanna_partition": args.rivanna_partition,
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Batch inputs/configuration changed; use a fresh output directory")
    atomic_json(config_path, config)
    completer = ChatAnnotationCompleter(
        model=args.model,
        max_tokens=args.max_tokens,
        thinking_effort=0.7,
        reports_dir=root / "api_calls",
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=args.provider_parameters,
    )
    accepted, records = [], []
    start = time.monotonic()
    try:
        for candidate in candidates:
            instance = KokkosInstance.from_dict(candidate)
            directory = root / instance.instance_id
            directory.mkdir(exist_ok=True)
            receipt_path = directory / "receipt.json"
            # Completed quarantine is cached too: repeating a command spends no new model budget.
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["config_sha256"] != digest(config):
                    raise ValueError("Stale receipt")
                if receipt["status"] == "accepted":
                    if any(
                        not (directory / name).is_file()
                        or hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha
                        for name, sha in receipt["evidence_hashes"].items()
                    ):
                        raise ValueError("Acceptance evidence changed")
                    accepted.append(json.loads((directory / "validated.json").read_text()))
                records.append(receipt)
                continue
            receipt = {
                "instance_id": instance.instance_id,
                "input_sha256": digest(candidate),
                "config_sha256": digest(config),
                "status": "quarantined",
                "stage": "exclusion",
                "execution_backend": "rivanna-slurm-apptainer",
            }
            began = time.monotonic()
            try:
                reason = exclusion(candidate, frozen, accepted)
                if reason:
                    raise ValueError(reason)
                if instance.metadata.get("test_origin") != "upstream":
                    raise ValueError(
                        "Model-generated/unknown test origin requires a separate validation lane"
                    )
                if (
                    instance.repo not in {"kokkos/kokkos", "kokkos/kokkos-kernels"}
                    or instance.metadata.get("accelerator") != "cpu"
                ):
                    raise ValueError("Unsupported Rivanna environment profile")
                if instance.repo == "kokkos/kokkos-kernels" and not re.fullmatch(
                    r"[0-9a-f]{40}", str(instance.metadata.get("kokkos_dependency_commit", ""))
                ):
                    raise ValueError("Historical kernels environment needs a pinned dependency")
                receipt["stage"] = "construction"
                atomic_json(directory / "progress.json", receipt)
                construction_path = directory / "construction.json"
                if construction_path.exists():
                    result = AutoAnnotationResult.from_dict(
                        json.loads(construction_path.read_text())
                    )
                else:
                    result = await annotate_and_validate_instance(
                        instance,
                        completer=completer,
                        max_attempts=args.max_attempts,
                        sandbox_timeout=1200,
                        command_timeout=300,
                        flaky_repetitions=3,
                        sandbox_factory=rivanna_validation_sandbox_factory,
                        validation_runner=bridge.validate,
                    )
                    atomic_json(construction_path, result.to_dict())
                if not result.passed or result.validated_instance is None:
                    last = result.attempts[-1] if result.attempts else None
                    raise ValueError(
                        (last.error or construction_failure(last.validation or {}))
                        if last
                        else "Missing construction evidence"
                    )
                instance = result.validated_instance
                if not instance.p2p_commands:
                    raise ValueError("Missing P2P regression coverage")
                receipt["stage"] = "instruction_audit"
                atomic_json(directory / "progress.json", receipt)
                from .mine import GitHubAPI
                from .revise_instructions import revise_and_audit

                instance, statement_passed = await revise_and_audit(
                    instance,
                    completer=completer,
                    api=GitHubAPI(os.environ.get("GITHUB_TOKEN")),
                    reports_dir=directory / "statement",
                    max_revisions=args.max_statement_revisions,
                    inference_config={
                        key: config[key]
                        for key in (
                            "model",
                            "base_url",
                            "provider_parameters",
                            "max_tokens",
                            "source_sha256",
                        )
                    },
                )
                if not statement_passed:
                    raise ValueError("Uncertain or unsupported statement after bounded revision")
                receipt["stage"] = "export"
                atomic_json(directory / "progress.json", receipt)
                task = export_instance(instance, directory / "export")
                receipt["export_sha256"] = tree_digest(task)
                receipt["stage"] = "independent_acceptance"
                evidence = await bridge.execute(instance, "acceptance")
                if evidence.get("error"):
                    raise RuntimeError(str(evidence["error"]))
                check_acceptance(evidence, receipt["export_sha256"])
                receipt["remote_acceptance"] = evidence
                receipt["trials"] = evidence["trials"]
                atomic_json(directory / "validated.json", instance.to_dict())
                receipt["status"] = "accepted"
                accepted.append(instance.to_dict())
            except Exception as error:
                receipt["failure_class"] = failure_class(str(error))
                # Redact configured secrets before saving error text.
                text = str(error)[-2000:]
                for key, secret in os.environ.items():
                    if ("KEY" in key or "TOKEN" in key or "SECRET" in key) and len(secret) > 8:
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
            write_rows(root / "accepted_training.jsonl", accepted)
            atomic_json(
                root / "report.json",
                {
                    "processed": len(records),
                    "accepted": len(accepted),
                    "automatic_acceptance_rate": len(accepted) / len(records),
                    "failure_classes": dict(
                        Counter(r.get("failure_class", "accepted") for r in records)
                    ),
                    "elapsed_seconds": round(time.monotonic() - start, 2),
                    "records": records,
                },
            )
            print(instance.instance_id, receipt["status"], receipt["stage"], flush=True)
        write_rows(root / "accepted_training.jsonl", accepted)
        usage = [
            json.loads(p.read_text()).get("response", {}).get("usage") or {}
            for p in (root / "api_calls").glob("*.json")
        ]
        atomic_json(
            root / "report.json",
            {
                "processed": len(records),
                "accepted": len(accepted),
                "automatic_acceptance_rate": len(accepted) / len(records) if records else 0,
                "failure_classes": dict(
                    Counter(r.get("failure_class", "accepted") for r in records)
                ),
                "model_calls": len(usage),
                "model_tokens": sum(u.get("total_tokens", 0) for u in usage),
                "monetary_cost": None,
                "cost_note": "Provider billing rate not supplied",
                "elapsed_seconds": round(time.monotonic() - start, 2),
                "records": records,
            },
        )
    finally:
        await completer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rivanna-host", required=True)
    parser.add_argument("--rivanna-root", required=True)
    parser.add_argument("--rivanna-base-sif", required=True)
    parser.add_argument(
        "--rivanna-partition", choices=("standard", "bii-largemem"), default="standard"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--frozen-instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--provider-parameters", type=json.loads, default={})
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-statement-revisions", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if (
        not 1 <= args.limit <= 10
        or not 1 <= args.max_attempts <= 3
        or not 1 <= args.max_tokens <= 16384
    ):
        parser.error("Require limit 1..10, attempts 1..3, max-tokens 1..16384")
    if not isinstance(args.provider_parameters, dict):
        parser.error("provider-parameters must be a JSON object")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
