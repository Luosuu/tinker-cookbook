"""Standalone runtime-test guard, also embedded in exported Harbor verifiers.

Do not import cookbook modules here: exported tasks execute this with system Python.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

COVERAGE_FAILURE = 86
ERROR_MARKER = "KOKKOS_RUNTIME_COVERAGE_ERROR"


def coverage_error(command: str, output: str, exit_code: int) -> str | None:
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    selected = [int(n) for n in re.findall(r"\[=+\]\s+Running (\d+) tests?", output)]
    passed = [int(n) for n in re.findall(r"\[\s*PASSED\s*\]\s+(\d+) tests?", output)]
    failed = [int(n) for n in re.findall(r"\[\s*FAILED\s*\]\s+(\d+) tests?", output)]
    ctest = [int(n) for n in re.findall(r"tests? failed out of\s+(\d+)", output)]
    if "No tests were found" in output or 0 in selected or 0 in ctest:
        return "A requested runtime test command selected zero tests"
    build_command = bool(re.match(r"\s*(?:cmake\s+--build|ninja|make)(?:\s|$)", command))
    google_test = (
        "--gtest_filter" in command
        or "GTEST_FILTER" in command
        or ("UnitTest" in command and not build_command)
    )
    ctest_command = bool(re.search(r"(?:^|[\s;&])ctest(?:\s|$)", command))
    if google_test and not selected:
        return "GoogleTest selection evidence is absent"
    if ctest_command and not ctest:
        return "CTest execution summary is absent"
    if exit_code == 0 and selected and (len(passed) != len(selected) or 0 in passed):
        return "GoogleTest reported success without a nonzero passed-test summary"
    if exit_code == 0 and (any(failed) or re.search(r"[1-9]\d* tests? failed out of", output)):
        return "Runtime test failures were masked by a successful shell exit"
    return None


def checked_shell(command: str) -> int:
    completed = subprocess.run(
        ["bash", "-lc", "set -o pipefail; " + command], text=True, capture_output=True
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    error = coverage_error(
        command, completed.stdout + "\n" + completed.stderr, completed.returncode
    )
    print(
        "KOKKOS_RUNTIME_COVERAGE "
        + json.dumps({"command": command, "exit_code": completed.returncode, "error": error}),
        flush=True,
    )
    if error:
        print(ERROR_MARKER + ": " + error, file=sys.stderr, flush=True)
        return COVERAGE_FAILURE
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(checked_shell(sys.argv[1]))
