"""Standalone runtime-test guard, also embedded in exported Harbor verifiers.

Do not import cookbook modules here: exported tasks execute this with system Python.
"""

from __future__ import annotations

import json
import fnmatch
import shlex
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


def _active_selected_gtest(command: str, output: str) -> str | None:
    """A crash must occur inside a named selected test, not before/after the suite."""
    try:
        args = shlex.split(command)
    except ValueError:
        return None
    selector = next((arg.split("=", 1)[1] for arg in args if arg.startswith("--gtest_filter=")), None)
    if selector is None and "--gtest_filter" in args:
        index = args.index("--gtest_filter")
        selector = args[index + 1] if index + 1 < len(args) else None
    if selector is None or not re.search(r"\[=+\]\s+Running [1-9]\d* tests?", output):
        return None
    active = None
    for match in re.finditer(r"\[\s*(RUN|OK|FAILED|SKIPPED)\s*\]\s+(\S+)", output):
        event, name = match.groups()
        if event == "RUN":
            active = name
        elif name == active:
            active = None
    if not active:
        return None
    positive, _, negative = selector.partition("-")
    if any(fnmatch.fnmatchcase(active, p) for p in (positive or "*").split(":")) and not any(
        fnmatch.fnmatchcase(active, p) for p in negative.split(":") if p
    ):
        return active
    return None


def classify_command_outcome(command: str, output: str, exit_code: int, runtime_check: bool) -> dict:
    """Classify evidence, not task intent. A failed test is never an automatic admission."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    def result(status: str, kind: str, error: str | None = None, **evidence) -> dict:
        return {"status": status, "kind": kind, "error": error, "raw_exit_code": exit_code, **evidence}
    for marker in ("No space left on device", "Cannot allocate memory", "Resource temporarily unavailable", "Killed signal terminated", "oom-kill", "Out of memory: Killed process", "std::bad_alloc", "out-of-memory", "memory allocation failed"):
        if marker.lower() in output.lower():
            return result("error", "resource_failure", marker)
    if exit_code in {124, 126, 127}:
        return result("error", "timeout_or_launcher_failure", "Reserved timeout/launcher exit; not test failure evidence")
    if "timed out" in output.lower():
        return result("error", "timeout", "Timeout diagnostic")
    if runtime_check:
        error = coverage_error(command, output, exit_code)
        if "no tests ran" in output.lower() or re.search(r"collected 0 items", output):
            error = "A requested runtime test command selected zero tests"
        if error:
            return result("error", "coverage_or_framework_failure", error)
    signal_number = -exit_code if exit_code < 0 else None
    # Positive shell signal encodings require a matching diagnostic; numbers alone are ambiguous.
    signatures = {6: r"\bAborted\b", 7: r"\bBus error\b", 8: r"\bFloating point exception\b", 11: r"\bSegmentation fault\b"}
    if 128 <= exit_code <= 192:
        candidate = exit_code - 128
        if candidate in signatures and re.search(signatures[candidate], output, re.I):
            signal_number = candidate
    if signal_number is not None:
        active = _active_selected_gtest(command, output) if runtime_check else None
        if signal_number in signatures and active:
            return result("failed", "target_program_signal", signal=signal_number, active_test=active)
        return result("error", "unknown_signal", "Signal provenance is not a supported active test failure", signal=signal_number)
    if exit_code >= 128:
        return result("error", "unknown_exit_encoding", "High exit status without trusted raw status or matching signal diagnostic")
    return result("passed" if exit_code == 0 else "failed", "success" if exit_code == 0 else "ordinary_nonzero")


def guarded_outcome(command: str, stdout: str, stderr: str, transport_code: int) -> dict:
    """Validate the guard's final envelope before recovering its raw child status."""
    prefix = "KOKKOS_RUNTIME_COVERAGE "
    lines = stdout.rstrip().splitlines()
    if not lines or not lines[-1].startswith(prefix):
        return classify_command_outcome(command, stdout + "\n" + stderr, transport_code, False)
    try:
        envelope = json.loads(lines[-1][len(prefix):])
        raw = envelope["exit_code"]
        if type(raw) is not int or envelope["command"] != command:
            raise ValueError("Guard command or raw exit identity mismatch")
        output = "\n".join(lines[:-1]) + "\n" + stderr
        outcome = classify_command_outcome(command, output, raw, True)
        allowed = {raw, raw % 256} if raw < 0 else {raw}
        if raw < 0:
            allowed.add(128 - raw)
        if outcome["status"] == "error" or envelope.get("error"):
            allowed.add(COVERAGE_FAILURE)
        if transport_code not in allowed:
            raise ValueError("Guard transport exit disagrees with raw exit")
        if envelope.get("error"):
            outcome.update(status="error", kind="coverage_or_framework_failure", error=envelope["error"])
        return outcome
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"status": "error", "kind": "invalid_guard_envelope", "error": str(error), "raw_exit_code": transport_code}


def checked_shell(command: str) -> int:
    completed = subprocess.run(
        ["bash", "-lc", "set -o pipefail; " + command], text=True, capture_output=True
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    outcome = classify_command_outcome(
        command, completed.stdout + "\n" + completed.stderr, completed.returncode, True
    )
    print(
        "KOKKOS_RUNTIME_COVERAGE "
        + json.dumps({"command": command, "exit_code": completed.returncode, "error": outcome["error"], "outcome": outcome}),
        flush=True,
    )
    if outcome["status"] == "error":
        print(ERROR_MARKER + ": " + str(outcome["error"]), file=sys.stderr, flush=True)
        return COVERAGE_FAILURE
    # Preserve raw status in the envelope and use the conventional shell signal form.
    return 128 - completed.returncode if completed.returncode < 0 else completed.returncode


if __name__ == "__main__":
    raise SystemExit(checked_shell(sys.argv[1]))
