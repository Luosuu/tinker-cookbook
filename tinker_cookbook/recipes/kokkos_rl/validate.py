"""Run the three-stage F2P/P2P validation for an annotated Kokkos instance."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.models import KokkosInstance


@dataclass(frozen=True)
class CommandResult:
    command: str
    expected_exit: str
    exit_code: int
    seconds: float
    stdout: str
    stderr: str

    @property
    def matched_expectation(self) -> bool:
        return self.exit_code == 0 if self.expected_exit == "zero" else self.exit_code != 0


@dataclass
class ValidationReport:
    instance_id: str
    passed: bool = False
    results: list[CommandResult] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "results": [asdict(item) for item in self.results],
            "error": self.error,
        }


def _run(
    command: str,
    *,
    cwd: Path,
    timeout: int,
    expected_exit: str = "zero",
    input_text: str | None = None,
) -> CommandResult:
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=command,
        expected_exit=expected_exit,
        exit_code=completed.returncode,
        seconds=time.monotonic() - started,
        stdout=completed.stdout[-16000:],
        stderr=completed.stderr[-16000:],
    )


def _record(report: ValidationReport, result: CommandResult) -> None:
    report.results.append(result)
    if not result.matched_expectation:
        raise RuntimeError(
            f"unexpected exit {result.exit_code} for {result.command!r}; "
            f"expected {result.expected_exit}"
        )


def _build_command(instance: KokkosInstance) -> str:
    targets = " ".join(instance.build_targets)
    return f"cmake --build build --target {targets} --parallel"


def validate_instance(
    instance: KokkosInstance,
    *,
    repo_dir: Path,
    command_timeout: int = 1200,
    flaky_repetitions: int = 3,
) -> ValidationReport:
    """Validate one instance in an incremental detached git worktree."""

    if not instance.is_validation_ready:
        raise ValueError(
            "instance needs build_targets, f2p_commands, test_patch, and code_patch before validation"
        )
    repo_dir = repo_dir.resolve()
    report = ValidationReport(instance_id=instance.instance_id)
    build_command = _build_command(instance)
    failure_stage = str(instance.metadata.get("f2p_stage", "test"))

    with tempfile.TemporaryDirectory(prefix=f"{instance.instance_id}-") as temp_dir:
        worktree = Path(temp_dir) / "repo"
        added = False
        try:
            add = _run(
                f"git worktree add --detach {worktree} {instance.base_commit}",
                cwd=repo_dir,
                timeout=command_timeout,
            )
            _record(report, add)
            added = True

            _record(
                report,
                _run(instance.configure_command, cwd=worktree, timeout=command_timeout),
            )
            _record(report, _run(build_command, cwd=worktree, timeout=command_timeout))
            for command in instance.p2p_commands:
                _record(report, _run(command, cwd=worktree, timeout=command_timeout))

            _record(
                report,
                _run(
                    "git apply --whitespace=nowarn -",
                    cwd=worktree,
                    timeout=command_timeout,
                    input_text=instance.test_patch,
                ),
            )
            if failure_stage == "build":
                _record(
                    report,
                    _run(
                        build_command,
                        cwd=worktree,
                        timeout=command_timeout,
                        expected_exit="nonzero",
                    ),
                )
            else:
                _record(report, _run(build_command, cwd=worktree, timeout=command_timeout))
                for _ in range(flaky_repetitions):
                    for command in instance.f2p_commands:
                        _record(
                            report,
                            _run(
                                command,
                                cwd=worktree,
                                timeout=command_timeout,
                                expected_exit="nonzero",
                            ),
                        )

            _record(
                report,
                _run(
                    "git apply --whitespace=nowarn -",
                    cwd=worktree,
                    timeout=command_timeout,
                    input_text=instance.code_patch,
                ),
            )
            _record(report, _run(build_command, cwd=worktree, timeout=command_timeout))
            for command in (*instance.f2p_commands, *instance.p2p_commands):
                _record(report, _run(command, cwd=worktree, timeout=command_timeout))
            report.passed = True
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            report.error = str(error)
        finally:
            if added:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=repo_dir,
                    capture_output=True,
                    check=False,
                )
    return report


def _load_instance(path: Path, instance_id: str | None) -> KokkosInstance:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if instance_id:
        rows = [row for row in rows if row.get("instance_id") == instance_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one instance, found {len(rows)}")
    return KokkosInstance.from_dict(rows[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-timeout", type=int, default=1200)
    parser.add_argument("--flaky-repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_instance(
        _load_instance(args.instances, args.instance_id),
        repo_dir=args.repo_dir,
        command_timeout=args.command_timeout,
        flaky_repetitions=args.flaky_repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"{report.instance_id}: {'PASS' if report.passed else 'FAIL'}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
