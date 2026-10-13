"""Compiler dependency evidence for test-header ownership; no object compilation."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# nvcc_wrapper forwards -M to the host compiler together with -o, which g++ rejects; such
# translation units are preprocessed instead and their line markers name every included file.
PREPROCESS_ONLY = {"nvcc_wrapper"}
LINE_MARKER = re.compile(r'^# \d+ "([^"<][^"]*)"', re.M)


def preprocess_arguments(arguments: list[str]) -> list[str]:
    return dependency_arguments(arguments, "")[:-5] + ["-E"]


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


# Executables that need another device backend or only time code cannot own a task.
NOT_TASK_TEST = {
    "cpu": re.compile(r"(?i)cuda|hip|sycl|openmptarget|openacc|hpx|threads|perf|benchmark"),
    "cuda": re.compile(r"(?i)hip|sycl|openmptarget|openacc|hpx|threads|perf|benchmark"),
}
NOT_HOST_TEST = NOT_TASK_TEST["cpu"]
# Test executables of the backend a task exercises, e.g. KokkosCore_UnitTest_Serial1.
PRIMARY_BACKEND = {"cpu": "Serial", "cuda": "Cuda"}
DISCOVERY_SECONDS = {"cpu": 150, "cuda": 600}


def host_test_executable(name: str, toolchain: str = "cpu") -> bool:
    return not NOT_TASK_TEST[toolchain].search(name)


def resolve_ownership(changed: list[str], observations: list[dict], failures: list[dict], primary: str = "Serial") -> dict:
    owners = {}
    for path in changed:
        found = sorted({row["target"] for row in observations if path in row["changed_dependencies"]})
        preferred = [target for target in found if primary.lower() in target.lower()]
        owners[path] = preferred or found
    unresolved = [path for path, targets in owners.items() if not targets]
    ambiguous = [path for path, targets in owners.items() if len(targets) != 1]
    targets = sorted({target for values in owners.values() for target in values})
    unique = targets[0] if len(targets) == 1 and not ambiguous and not failures else None
    return {"target": unique, "owners": owners, "unresolved": unresolved, "ambiguous": ambiguous,
            "observations": observations, "failures": failures, "method": "compiler_preprocessor_dependencies", "objects_compiled": 0}


def discover(root: Path, changed: list[str], max_seconds: int | None = None, toolchain: str = "cpu") -> dict:
    """Primary-backend executables first; other eligible ones only for files those do not own."""
    primary = PRIMARY_BACKEND[toolchain]
    # CUDA translation units are preprocessed by nvcc, several times slower than g++ -M.
    workers = 8 if toolchain == "cuda" else 1
    if max_seconds is None:
        max_seconds = DISCOVERY_SECONDS[toolchain]
    root = root.resolve()
    models = [json.loads(p.read_text()) for p in (root / "build/.cmake/api/v1/reply").glob("target-*.json")]
    executables = [m for m in models if m.get("type") == "EXECUTABLE"]
    commands = json.loads((root / "build/compile_commands.json").read_text())
    observations, failures = [], []
    start = time.monotonic()
    first = [m for m in executables if primary in m["name"]]
    if _observe(root, first, commands, changed, observations, failures, start, max_seconds, workers):
        return resolve_ownership(changed, observations, failures, primary)
    unowned = [path for path in changed if not any(path in row["changed_dependencies"] for row in observations)]
    if unowned and not failures:
        second = [m for m in executables if primary not in m["name"] and host_test_executable(m["name"], toolchain)]
        _observe(root, second, commands, unowned, observations, failures, start, max_seconds, workers)
    return resolve_ownership(changed, observations, failures, primary)


def _observe(root, models, commands, changed, observations, failures, start, max_seconds, workers=1) -> bool:
    """Record changed dependencies per translation unit; True when the time budget ran out."""
    jobs = []
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
            jobs.append((root, target["name"], source, directory, arguments, changed, start + max_seconds))
    if workers == 1:
        for job in jobs:
            kind, row = _translation_unit(*job)
            if kind == "budget":
                failures.append(row)
                return True
            (observations if kind == "observed" else failures).append(row)
        return False
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda job: _translation_unit(*job), jobs))
    budget = [row for kind, row in results if kind == "budget"]
    observations.extend(row for kind, row in results if kind == "observed")
    failures.extend(row for kind, row in results if kind == "failed")
    failures.extend(budget[:1])
    return bool(budget)


def _translation_unit(root, target, source, directory, arguments, changed, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "budget", {"target": target, "error": "dependency_discovery_time_budget"}
    unit = str(source.relative_to(root))
    try:
        if Path(arguments[0]).name in PREPROCESS_ONLY:
            completed = subprocess.run(preprocess_arguments(arguments), cwd=directory, capture_output=True, text=True, timeout=min(60, remaining))
            if completed.returncode:
                return "failed", {"target": target, "translation_unit": unit, "exit_code": completed.returncode, "stderr": completed.stderr[-2000:]}
            dependencies = {(directory / value).resolve() for value in LINE_MARKER.findall(completed.stdout)}
        else:
            with tempfile.TemporaryDirectory(prefix="kokkos-dependencies-") as temporary:
                dep = Path(temporary) / "deps.d"
                completed = subprocess.run(dependency_arguments(arguments, str(dep)), cwd=directory, capture_output=True, text=True, timeout=min(20, remaining))
                if completed.returncode or not dep.exists():
                    return "failed", {"target": target, "translation_unit": unit, "exit_code": completed.returncode, "stderr": completed.stderr[-2000:]}
                text = dep.read_text().replace("\\\n", " ").partition(":")[2]
                dependencies = {(directory / value).resolve() for value in shlex.split(text)}
    except subprocess.TimeoutExpired:
        return "failed", {"target": target, "translation_unit": unit, "error": "dependency_command_timeout"}
    matches = [path for path in changed if (root / path).resolve() in dependencies]
    return "observed", {"target": target, "translation_unit": unit, "changed_dependencies": matches, "exit_code": 0}


if __name__ == "__main__":
    import sys
    toolchain = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    print(json.dumps(discover(Path(sys.argv[1]), json.loads(sys.argv[2]), toolchain=toolchain)))
