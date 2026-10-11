import pytest

from .expansion_batch import failure_class, source_digest
from .rivanna_bridge import RivannaBridge


def test_source_identity_and_failure_buckets():
    assert len(source_digest()) == 64
    assert failure_class("TimeoutError: command timed out") == "timeout"
    assert failure_class("Apptainer startup failed") == "environment_error"
    assert failure_class("Assertion mismatch") == "invalid_or_unresolved_task"


def test_bridge_refuses_shell_injection(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        RivannaBridge(
            tmp_path, host="host; echo bad", remote_root="/scratch/run", source_sha256="x"
        )


@pytest.mark.asyncio
async def test_rivanna_factory_rejects_local_execution(monkeypatch):
    from .rivanna_validate import rivanna_validation_sandbox_factory

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Rivanna"):
        await rivanna_validation_sandbox_factory(None, 60)


def test_historical_kernels_source_layout():
    from .patching import is_source_path, is_test_path

    assert is_source_path("src/sparse/KokkosSparse_CrsMatrix.hpp", "kokkos/kokkos-kernels")
    assert is_test_path("unit_test/sparse/Test_Sparse_spmv.hpp", "kokkos/kokkos-kernels")


def test_final_gate_rejects_partial_trials_and_wrong_rewards():
    from .expansion_batch import check_acceptance

    evidence = {
        "export_sha256": "export",
        "sif_sha256": "image",
        "job_id": "1",
        "trials": [
            {"mode": m, "reward": int(m == "oracle"), "sha256": "proof", "isolation_probe": True}
            for m in ["nop", "oracle", "nop", "oracle", "tampered-nop"]
        ],
    }
    check_acceptance(evidence, "export")
    with pytest.raises(ValueError, match="Incomplete"):
        check_acceptance({**evidence, "trials": evidence["trials"][:2]}, "export")
    evidence["trials"][-1]["reward"] = 1
    with pytest.raises(ValueError, match="Unstable"):
        check_acceptance(evidence, "export")


def test_flakiness_is_scoped_to_repository_state():
    from .expansion_batch import construction_failure

    def row(command, code, expected="zero"):
        return {"command": command, "exit_code": code, "expected_exit": expected}

    report = {
        "error": "mismatch",
        "results": [
            row("test", 0),
            row("git apply /tmp/kokkos-test.patch", 0),
            row("test", 1, "nonzero"),
        ],
    }
    assert construction_failure(report) == "mismatch"
    report["results"].append(row("test", 0, "nonzero"))
    assert construction_failure(report).startswith("unstable:")
    report["results"][-1]["exit_code"] = 124
    assert construction_failure(report).startswith("timeout:")


def test_kernels_export_uses_explicit_immutable_dependency():
    from dataclasses import replace

    from .export_harbor import _dockerfile
    from .models_test import _instance

    commit = "b5b7660af56d883ffc8892fa8a253ee257f38e26"
    instance = replace(
        _instance(),
        repo="kokkos/kokkos-kernels",
        metadata={"toolchain": "cpu", "kokkos_dependency_commit": commit},
    )
    dockerfile = _dockerfile(instance)
    assert "fetch --depth 1 origin " + commit in dockerfile
    assert "git clone --branch 5.2.0" not in dockerfile


@pytest.mark.asyncio
async def test_kernels_requires_immutable_dependency_before_execution(monkeypatch):
    from dataclasses import replace

    from .models_test import _instance
    from .rivanna_validate import rivanna_validation_sandbox_factory

    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "shen")
    instance = replace(_instance(), repo="kokkos/kokkos-kernels", metadata={"toolchain": "cpu"})
    with pytest.raises(ValueError, match="immutable"):
        await rivanna_validation_sandbox_factory(instance, 60)


def test_adapter_negative_timeout_is_classified_separately():
    from .expansion_batch import construction_failure

    report = {
        "error": "unexpected exit -1",
        "results": [
            {
                "command": "cmake --build build",
                "exit_code": -1,
                "expected_exit": "zero",
                "stderr": "Command timed out after 300 seconds",
            }
        ],
    }
    assert construction_failure(report).startswith("timeout:")
    report["results"][0]["stderr"] = "container transport failed"
    assert construction_failure(report).startswith("environment_error:")


def test_negative_trials_require_causal_failure_and_integrity_evidence():
    from dataclasses import replace
    from copy import deepcopy
    from .models_test import _instance
    from .expansion_batch import validate_trial_checks

    instance = replace(_instance(), f2p_commands=("compile",), p2p_commands=("test",), metadata={"f2p_stage": "build"})
    checks = [
        {"id": "setup", "status": "passed", "exit_code": 0},
        {"id": "build", "status": "failed", "exit_code": 2},
        {"id": "f2p-0", "status": "blocked", "blocked_by": "build"},
        {"id": "p2p-0", "status": "blocked", "blocked_by": "build"},
    ]
    log = {"diagnostics": {"complete": True, "checks": checks}, "stdout": "error: no matching function"}
    assert validate_trial_checks(instance, "nop", log) == "compile_regression_observed"
    with pytest.raises(ValueError):
        validate_trial_checks(instance, "oracle", log)
    with pytest.raises(ValueError):
        validate_trial_checks(instance, "tampered-nop", log)
    for mutation in ("missing", "timeout", "uncausal", "setup", "no_compiler_error"):
        bad = deepcopy(log)
        rows = bad["diagnostics"]["checks"]
        if mutation == "missing": rows.pop()
        if mutation == "timeout": rows[1]["exit_code"] = 124
        if mutation == "uncausal": rows[2]["blocked_by"] = "unknown"
        if mutation == "setup": rows[0]["status"] = "failed"; rows[0]["exit_code"] = 1
        if mutation == "no_compiler_error": bad["stdout"] = "build failed"
        with pytest.raises((ValueError, RuntimeError)):
            validate_trial_checks(instance, "nop", bad)
    checks[0].update(status="failed", exit_code=1)
    for row in checks[1:]: row.update(status="blocked", blocked_by="setup", exit_code=None)
    log["stdout"] = "candidate patch changes protected test/build/CI files"
    assert validate_trial_checks(instance, "tampered-nop", log) == "protected_edit_rejected"
    with pytest.raises(ValueError):
        validate_trial_checks(instance, "nop", log)
    log["stdout"] = "failed to inject hidden tests"
    with pytest.raises(ValueError):
        validate_trial_checks(instance, "tampered-nop", log)
