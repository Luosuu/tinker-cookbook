"""Model-free CPU target and runtime inventory discovery on Rivanna."""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
from pathlib import Path

from .models import KokkosInstance


def gtest_names(output: str) -> list[str]:
    names = []
    suite = ""
    for line in output.splitlines():
        text = line.split("#", 1)[0].rstrip()
        if text and not text[0].isspace() and text.endswith("."):
            suite = text.strip()
        elif suite and text.startswith("  ") and text.strip():
            names.append(suite + text.strip())
    return sorted(set(names))


def selected_tests(names: list[str], selector: str) -> list[str]:
    positive, _, negative = selector.partition("-")
    included = positive.split(":") if positive else ["*"]
    excluded = negative.split(":") if negative else []
    return [
        name
        for name in names
        if any(fnmatch.fnmatchcase(name, p) for p in included)
        and not any(fnmatch.fnmatchcase(name, p) for p in excluded)
    ]


def require_baseline_selection(instance: KokkosInstance) -> None:
    inventory = instance.metadata.get("runtime_test_inventory")
    if not isinstance(inventory, dict):
        return
    names = inventory.get("test_names", [])
    target = inventory.get("target", "")
    if not names or not target or not instance.p2p_commands:
        raise ValueError("Preflight inventory/P2P selection is empty")
    for command in instance.p2p_commands:
        match = re.search(r"--gtest_filter(?:=|\s+)(?:'([^']*)'|\"([^\"]*)\"|(\S+))", command)
        if target not in command or not match:
            raise ValueError("P2P command is not bound to the discovered target and selector")
        selector = next(part for part in match.groups() if part is not None)
        if not selected_tests(names, selector):
            raise ValueError("P2P selector matches no discovered runtime tests")


async def inventory(instance: KokkosInstance, *, discovery_only: bool = False) -> dict:
    from .rivanna_validate import rivanna_validation_sandbox_factory

    if discovery_only and not any(
        "test" in file.filename.lower()
        and Path(file.filename).suffix in {".h", ".hpp", ".hh", ".hxx", ".c", ".cpp", ".cc", ".cxx"}
        for file in instance.changed_files
    ):
        return {"passed": False, "stage": "unsupported_discovery_input", "runtime_tests_executed": 0, "target_builds": 0}

    from .qualification import execution_policy
    policy = execution_policy(instance)
    sandbox = await rivanna_validation_sandbox_factory(instance, policy["sandbox_seconds"])
    records = []

    async def run(command, timeout=60):
        result = await sandbox.run_command(command, timeout=timeout)
        records.append(
            {
                "command": command,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result

    try:
        await run(
            "mkdir -p build/.cmake/api/v1/query && touch build/.cmake/api/v1/query/codemodel-v2"
        )
        result = await run(instance.configure_command + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON", 180)
        if result.exit_code:
            return {"passed": False, "stage": "configure", "records": records}
        target_result = await run("cmake --build build --target help")
        ctest_result = await run("ctest --test-dir build --show-only=json-v1")
        if target_result.exit_code or ctest_result.exit_code:
            return {"passed": False, "stage": "inventory", "records": records}
        targets = re.findall(r"(?m)^([^\s:]+):", target_result.stdout)
        ctest = json.loads(ctest_result.stdout)
        from .target_discovery import NOT_TASK_TEST, PRIMARY_BACKEND

        toolchain = "cuda" if instance.metadata.get("toolchain") == "cuda" else "cpu"
        primary = PRIMARY_BACKEND[toolchain]
        codemodel_result = await run(
            "python3 -c "
            + shlex.quote(
                "import glob,json,re; ts=[json.load(open(p)) for p in glob.glob('build/.cmake/api/v1/reply/target-*.json')]; print(json.dumps([{'name':t['name'],'type':t['type'],'artifacts':t.get('artifacts',[]),'sources':[{'path':s['path']} for s in t.get('sources',[])]} for t in ts if t.get('type')=='EXECUTABLE' and not re.search("
                + repr(NOT_TASK_TEST[toolchain].pattern)
                + ", t['name'])]))"
            )
        )
        codemodel = json.loads(codemodel_result.stdout) if codemodel_result.exit_code == 0 else []
        requested = instance.metadata.get("preflight_target")
        executables = {
            t["name"]: "/workspace/repo/build/" + t["artifacts"][0]["path"]
            for t in codemodel
            if t.get("type") == "EXECUTABLE" and t.get("artifacts")
        }
        changed = {
            file.filename for file in instance.changed_files if "test" in file.filename.lower()
        }
        owners = [
            t["name"]
            for t in codemodel
            if primary in t["name"]
            and changed.intersection(source["path"] for source in t.get("sources", []))
        ]
        discovery = None
        test_sources = sorted(path for path in changed if Path(path).suffix in {".h", ".hpp", ".hh", ".hxx", ".c", ".cpp", ".cc", ".cxx"})
        # Direct sources alone cannot establish ownership of headers or newly added tests.
        if test_sources:
            from . import target_discovery

            written = await sandbox.write_file("/tmp/kokkos-target-discovery.py", Path(target_discovery.__file__).read_bytes())
            if written.exit_code:
                return {"passed": False, "stage": "discovery_upload", "records": records}
            # Use the test overlay only to discover its actual include relationships.
            # It is reversed before any baseline build or runtime inventory.
            await sandbox.write_file("/tmp/kokkos-discovery-tests.patch", instance.test_patch)
            applied = await run("git apply --whitespace=nowarn /tmp/kokkos-discovery-tests.patch")
            if applied.exit_code:
                return {"passed": False, "stage": "discovery_patch", "records": records}
            configured = await run(instance.configure_command + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON", 180)
            if configured.exit_code == 0:
                observed = await run("python3 /tmp/kokkos-target-discovery.py /workspace/repo " + shlex.quote(json.dumps(test_sources)) + " " + toolchain, 180)
                if observed.exit_code == 0:
                    discovery = json.loads(observed.stdout)
            restored = await run("git apply -R --whitespace=nowarn /tmp/kokkos-discovery-tests.patch")
            if restored.exit_code:
                return {"passed": False, "stage": "discovery_restore", "records": records}
            if discovery_only:
                return {"passed": bool(discovery and discovery.get("target") in targets), "stage": "target_discovery_only", "discovery": discovery, "records": records, "runtime_tests_executed": 0, "target_builds": 0}
            restored_config = await run(instance.configure_command + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON", 180)
            if restored_config.exit_code:
                return {"passed": False, "stage": "baseline_reconfigure", "records": records}
            # Hints cannot override missing, failed, or ambiguous dependency evidence.
            requested = discovery.get("target") if discovery else None
        elif requested is None and len(owners) == 1:
            requested = owners[0]
        if requested not in targets or requested not in executables:
            return {
                "passed": False,
                "stage": "target_resolution",
                "discovery": discovery,
                "targets": targets,
                "ctest": ctest,
                "codemodel": codemodel,
                "records": records,
            }
        target = shlex.quote(requested)
        await run("ninja -C build -t commands " + target + " | wc -l")
        result = await run("cmake --build build --target " + target + f" --parallel {policy['build_parallelism']}", policy["build_seconds"])
        if result.exit_code:
            return {"passed": False, "stage": "build", "target": requested, "records": records}
        binary = executables[requested]
        result = await run(shlex.quote(binary) + " --gtest_list_tests")
        names = gtest_names(result.stdout) if result.exit_code == 0 else []
        return {
            "passed": bool(names),
            "stage": "runtime_inventory",
            "discovery": discovery,
            "target": requested,
            "binary": binary,
            "test_names": names,
            "ctest": ctest,
            "codemodel": codemodel,
            "records": records,
        }
    finally:
        await sandbox.cleanup()
