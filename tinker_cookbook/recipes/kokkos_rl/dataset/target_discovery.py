"""Compiler dependency evidence for test-header ownership; no object compilation."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


def dependency_arguments(arguments: list[str], dependency_file: str) -> list[str]:
    result = []
    skip = False
    for word in arguments:
        if skip:
            skip = False
            continue
        if word in {"-o", "-MF", "-MT", "-MQ"}:
            skip = True
        elif word in {"-c", "-MD", "-MMD", "-MP"}:
            continue
        elif any(word.startswith(prefix) and word != prefix for prefix in ("-MF", "-MT", "-MQ")):
            continue
        else:
            result.append(word)
    return result + ["-M", "-MF", dependency_file, "-MT", "discovery"]


# Executables that need a device backend or only time code cannot own a CPU task.
NOT_HOST_TEST = re.compile(r"(?i)cuda|hip|sycl|openmptarget|openacc|hpx|threads|perf|benchmark")


def host_test_executable(name: str) -> bool:
    return not NOT_HOST_TEST.search(name)


def resolve_ownership(changed: list[str], observations: list[dict], failures: list[dict]) -> dict:
    owners = {}
    for path in changed:
        found = sorted({row["target"] for row in observations if path in row["changed_dependencies"]})
        serial = [target for target in found if "serial" in target.lower()]
        owners[path] = serial or found
    unresolved = [path for path, targets in owners.items() if not targets]
    ambiguous = [path for path, targets in owners.items() if len(targets) != 1]
    targets = sorted({target for values in owners.values() for target in values})
    unique = targets[0] if len(targets) == 1 and not ambiguous and not failures else None
    return {"target": unique, "owners": owners, "unresolved": unresolved, "ambiguous": ambiguous,
            "observations": observations, "failures": failures, "method": "compiler_preprocessor_dependencies", "objects_compiled": 0}


def discover(root: Path, changed: list[str], max_seconds: int = 150) -> dict:
    """Serial executables first; other host test executables only for files no Serial one owns."""
    root = root.resolve()
    models = [json.loads(p.read_text()) for p in (root / "build/.cmake/api/v1/reply").glob("target-*.json")]
    executables = [m for m in models if m.get("type") == "EXECUTABLE"]
    commands = json.loads((root / "build/compile_commands.json").read_text())
    observations, failures = [], []
    start = time.monotonic()
    first = [m for m in executables if "Serial" in m["name"]]
    if _observe(root, first, commands, changed, observations, failures, start, max_seconds):
        return resolve_ownership(changed, observations, failures)
    unowned = [path for path in changed if not any(path in row["changed_dependencies"] for row in observations)]
    if unowned and not failures:
        second = [m for m in executables if "Serial" not in m["name"] and host_test_executable(m["name"])]
        _observe(root, second, commands, unowned, observations, failures, start, max_seconds)
    return resolve_ownership(changed, observations, failures)


def _observe(root, models, commands, changed, observations, failures, start, max_seconds) -> bool:
    """Record changed dependencies per translation unit; True when the time budget ran out."""
    for target in models:
        sources = {(root / s["path"]).resolve() for s in target.get("sources", [])}
        for command in commands:
            directory = Path(command["directory"])
            source = (directory / command["file"]).resolve()
            if source not in sources:
                continue
            arguments = command.get("arguments") or shlex.split(command["command"])
            # The actual object path binds a command to its CMake target when a TU is shared.
            output = next((arguments[i + 1] for i, a in enumerate(arguments[:-1]) if a == "-o"), "")
            if f"CMakeFiles/{target['name']}.dir/" not in output.replace(os.sep, "/"):
                continue
            remaining = max_seconds - (time.monotonic() - start)
            if remaining <= 0:
                failures.append({"target": target["name"], "error": "dependency_discovery_time_budget"})
                return True
            with tempfile.TemporaryDirectory(prefix="kokkos-dependencies-") as temporary:
                dep = Path(temporary) / "deps.d"
                argv = dependency_arguments(arguments, str(dep))
                try:
                    completed = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=min(20, remaining))
                    if completed.returncode or not dep.exists():
                        failures.append({"target": target["name"], "translation_unit": str(source.relative_to(root)), "exit_code": completed.returncode, "stderr": completed.stderr[-2000:]})
                        continue
                    text = dep.read_text().replace("\\\n", " ").partition(":")[2]
                    dependencies = {(directory / value).resolve() for value in shlex.split(text)}
                    matches = [path for path in changed if (root / path).resolve() in dependencies]
                    observations.append({"target": target["name"], "translation_unit": str(source.relative_to(root)), "changed_dependencies": matches, "exit_code": 0})
                except subprocess.TimeoutExpired:
                    failures.append({"target": target["name"], "translation_unit": str(source.relative_to(root)), "error": "dependency_command_timeout"})
    return False


if __name__ == "__main__":
    import sys
    print(json.dumps(discover(Path(sys.argv[1]), json.loads(sys.argv[2]))))
