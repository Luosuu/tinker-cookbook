"""Standalone command-level verifier reporting; embedded in exported scripts."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path


def run_suite(
    plan: dict[str, object],
    coverage_check: Callable[[str, str, int], str | None],
) -> None:
    directory = Path("/logs/verifier")
    directory.mkdir(parents=True, exist_ok=True)
    reward = directory / "reward.txt"
    reward.write_text("0\n")
    checks = plan["checks"]
    assert isinstance(checks, list)
    results: list[dict[str, object]] = []
    blocked_by: str | None = None
    for item in checks:
        assert isinstance(item, dict)
        identifier, stage, command = str(item["id"]), str(item["stage"]), str(item["command"])
        row: dict[str, object] = {"id": identifier, "stage": stage, "command": command}
        if blocked_by is not None:
            row.update(status="blocked", blocked_by=blocked_by, exit_code=None, elapsed_s=0)
        elif not command.strip():
            row.update(status="error", error="Empty required command", exit_code=None, elapsed_s=0)
        else:
            start = time.monotonic()
            try:
                completed = subprocess.run(
                    ["bash", "-c", "set -o pipefail; " + command],
                    cwd="/workspace/repo",
                    text=True,
                    capture_output=True,
                )
                output = completed.stdout + "\n" + completed.stderr
                error = (
                    coverage_check(command, output, completed.returncode)
                    if stage in {"f2p", "p2p"}
                    else None
                )
                if stage in {"f2p", "p2p"} and (
                    "no tests ran" in output or re.search(r"collected 0 items", output)
                ):
                    error = "A requested runtime test command selected zero tests"
                infrastructure = (
                    completed.returncode < 0
                    or completed.returncode in {126, 127, 137, 143}
                    or any(
                        marker in output
                        for marker in [
                            "No space left on device",
                            "Cannot allocate memory",
                            "Resource temporarily unavailable",
                            "Killed signal terminated",
                        ]
                    )
                )
                status = (
                    "error"
                    if error or infrastructure
                    else ("passed" if completed.returncode == 0 else "failed")
                )
                row.update(
                    status=status,
                    exit_code=completed.returncode,
                    error=error or ("Process or resource failure" if infrastructure else None),
                    elapsed_s=round(time.monotonic() - start, 6),
                )
                (directory / (identifier + ".stdout")).write_text(completed.stdout)
                (directory / (identifier + ".stderr")).write_text(completed.stderr)
                print(completed.stdout, end="", flush=True)
                print(completed.stderr, end="", flush=True)
            except OSError as error:
                row.update(
                    status="error",
                    error=str(error),
                    exit_code=None,
                    elapsed_s=round(time.monotonic() - start, 6),
                )
        results.append(row)
        if (
            stage in {"setup", "build"}
            or re.match(r"\s*(?:cmake\s+--build|ninja|make)(?:\s|$)", command)
        ) and row["status"] not in {"passed", "blocked"}:
            blocked_by = identifier
        report = {
            "schema_version": 1,
            "task": plan["task"],
            "base_commit": plan["base_commit"],
            "test_patch_sha256": plan["test_patch_sha256"],
            "verifier_sha256": plan["verifier_sha256"],
            "count_unit": "commands",
            "complete": len(results) == len(checks),
            "checks": results,
            "reward": int(
                len(results) == len(checks) and all(r["status"] == "passed" for r in results)
            ),
        }
        temporary = directory / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(directory / "results.json")
    reward.write_text(str(report["reward"]) + "\n")
