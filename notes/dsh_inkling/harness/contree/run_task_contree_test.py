"""Guards for the harness profile and bridge plugin.

Two failures in this project were silent rather than loud, and each invalidated a full
100-task run: a plugin whose config was rejected simply did not load, and a half-applied
edit left the bridge referencing an undefined variable, so no tool result ever reached the
harness and every episode hung forever after its first request. These tests make both
classes fail fast instead of surfacing as an unexplained score.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
PROFILE = HERE / "dsh.patch.yml.in"
BRIDGE = HERE / "sandbox-bridge.mjs"
RUNTIME = Path(__file__).parents[4] / ".dsh-runtime"

# The pruner rejects configurations where head + marker + tail exceeds the threshold.
# Measured from the installed plugin; the test below re-derives it.
PRUNE_MARKER_CODE_POINTS = 39


def _entries() -> list[dict]:
    text = PROFILE.read_text().replace("${DSH_PERSONA}", "persona")
    return yaml.safe_load(text)


# --- profile ------------------------------------------------------------------------


def test_profile_parses_as_a_patch_list() -> None:
    entries = _entries()
    assert isinstance(entries, list) and entries
    assert all(isinstance(entry, dict) for entry in entries)


def test_pruner_budget_is_accepted_by_the_plugin() -> None:
    """A rejected pruner config disables the plugin without failing the run."""
    entry = next(e for e in _entries() if e.get("id") == "tool-result-pruner")
    config = entry["config"]
    emitted = config["headChars"] + PRUNE_MARKER_CODE_POINTS + config["tailChars"]
    assert emitted <= config["thresholdChars"], (
        f"pruner would be rejected: {emitted} > {config['thresholdChars']}"
    )


@pytest.mark.skipif(
    not (RUNTIME / "node_modules/@deepseek-ai/dsh-compaction-tool-result-pruner").is_dir(),
    reason="DSH runtime not installed locally",
)
def test_pruner_marker_length_still_matches_the_constant() -> None:
    """The guard above is only valid while the marker is the length we measured."""
    source = (
        RUNTIME / "node_modules/@deepseek-ai/dsh-compaction-tool-result-pruner/lib/index.js"
    ).read_text()
    match = re.search(r'PRUNE_MARKER = "((?:[^"\\]|\\.)*)"', source)
    assert match, "PRUNE_MARKER not found; the plugin layout changed"
    marker = match.group(1).encode().decode("unicode_escape")
    assert len(marker) == PRUNE_MARKER_CODE_POINTS, (
        f"marker is now {len(marker)} code points; update the guard and the profile"
    )


def test_reference_inference_parameters_are_not_set_in_the_profile() -> None:
    """The provider cannot express these; the proxy injects them, by construction."""
    assert "reasoning_effort" not in PROFILE.read_text()


def test_prompt_matching_plugins_are_disabled() -> None:
    disabled = {e["id"] for e in _entries() if e.get("disabled") is True}
    for entry_id in ("plan-mode", "agent-instructions", "session-title-llm", "session-title"):
        assert entry_id in disabled, f"{entry_id} must stay disabled for comparability"


def test_exactly_one_tool_is_inserted() -> None:
    """The reference exposes one repository tool; a second changes two variables."""
    inserts = [e for e in _entries() if "insert" in e]
    assert len(inserts) == 1
    assert len(inserts[0]["insert"]) == 1


# --- bridge plugin ------------------------------------------------------------------


def _run_bridge_against_stub(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Load and execute the real bridge with the tool SDK stubbed.

    The bridge imports `@deepseek-ai/dsh-tools` by bare specifier, so a stub is written
    into a temporary node_modules beside a copy of the plugin. Executing the tool is what
    catches an undefined identifier, which a syntax check cannot.
    """
    root = tmp_path / "bridge"
    (root / "node_modules/@deepseek-ai/dsh-tools").mkdir(parents=True)
    (root / "node_modules/@deepseek-ai/dsh-tools/index.js").write_text(
        "export const defineTool = (tool) => tool;\n"
    )
    (root / "node_modules/@deepseek-ai/dsh-tools/package.json").write_text(
        json.dumps({"name": "@deepseek-ai/dsh-tools", "type": "module", "main": "index.js"})
    )
    (root / "package.json").write_text(json.dumps({"type": "module"}))
    (root / "bridge.mjs").write_text(BRIDGE.read_text())
    (root / "drive.mjs").write_text(
        """
import { createServer } from 'node:net';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const tools = [];
const plugin = await import(pathToFileURL(join(import.meta.dirname, 'bridge.mjs')).href);
plugin.apply({ tools: { register: (tool) => tools.push(tool) } });
if (tools.length !== 1) throw new Error(`expected one tool, got ${tools.length}`);

const dir = mkdtempSync(join(tmpdir(), 'bridge-'));
const socketPath = join(dir, 'rpc.sock');
const server = createServer((connection) => {
  connection.on('data', () => {
    connection.write(JSON.stringify({ exit_code: 0, stdout: 'STUB_OK', stderr: '' }) + '\\n');
  });
});
await new Promise((resolve) => server.listen(socketPath, resolve));
process.env.HPC_BRIDGE_SOCKET = socketPath;

const value = await tools[0].execute({ command: 'ls /workspace/repo' }, {});
if (typeof value !== 'string' || !value.includes('STUB_OK')) {
  throw new Error(`unexpected bridge output: ${value}`);
}
console.log('BRIDGE_OK');
server.close();
"""
    )
    return subprocess.run(
        ["node", "drive.mjs"], cwd=str(root), capture_output=True, text=True, timeout=20
    )


def test_bridge_executes_and_returns_a_tool_result(tmp_path: Path) -> None:
    """An undefined identifier here hangs every episode after its first request."""
    result = _run_bridge_against_stub(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "BRIDGE_OK" in result.stdout, result.stdout


# From eval_kokkos_openai.BASH_TOOL, which is the tool the audited 52/100 baseline used.
REFERENCE_TOOL_DESCRIPTION = "Execute a bash command in the sandbox environment."


def test_bridge_matches_the_reference_tool_spec() -> None:
    """The tool schema the model sees must match the reference's, byte for byte.

    The first version described the tool as a way to "inspect, edit, compile, and debug",
    which is not what the reference harness said. Since the tool description is part of the
    prompt, a difference here changes the comparison without being visible in the score.
    """
    source = BRIDGE.read_text()
    assert json.dumps(REFERENCE_TOOL_DESCRIPTION) in source, (
        "bridge tool description drifted from the reference tool spec"
    )
    # The reference tool takes a bare string parameter with no description.
    assert "{ type: 'string', required: true }" in source


def test_cpu_affinity_default_is_two() -> None:
    """Four CPUs makes the verifier's parallel build OOM on a 4 GB ConTree sandbox.

    Measured: 4 CPUs builds a small target faster, which is why it was briefly the
    default, but the exported verifier's full `--parallel` build then dies with
    `cc1plus: Killed signal terminated` and scores correct patches as zero. A run
    configured that way showed 12 grading OOMs against 2 for the 2-CPU configuration.
    """
    import inspect

    from notes.dsh_inkling.harness.contree import run_batch, run_task_contree

    for module in (run_batch, run_task_contree):
        source = inspect.getsource(module)
        assert 'default="0,1"' in source, (
            f"{module.__name__} must pin to two CPUs; four OOMs the verifier"
        )
        assert 'default="0,1,2,3"' not in source


def test_reported_budget_is_the_binding_request_limit() -> None:
    """The bridge must quote the request cap, not the tool-call cap.

    Across 700 episodes no run reached the 80 tool-call cap while 453 hit the 40-request
    cap, and the two counts track at 0.99 calls per request. Quoting 80 therefore
    understates consumption by half.
    """
    source = BRIDGE.read_text()
    assert "HPC_MAX_REQUESTS" in source
    assert "model requests" in source
    # The old wording quoted the tool-call cap.
    assert "Budget spent: ${calls}/${MAX_TOOL_CALLS}" not in source


def test_only_build_pipelines_are_wrapped_with_pipefail() -> None:
    """Build output pagers must not hide failure, without breaking read pipelines."""
    from notes.dsh_inkling.harness.contree.run_task_contree import (
        _with_truthful_build_status,
    )

    build = "cd build && ninja Kokkos_CoreUnitTest_Serial1 2>&1 | tail -n 40"
    wrapped = _with_truthful_build_status(build)
    assert wrapped.startswith("bash -o pipefail -c ")
    assert build in shlex.split(wrapped)

    synthetic = "ninja() { return 7; }; ninja 2>&1 | tail -n 1"
    assert subprocess.run(["bash", "-c", synthetic]).returncode == 0
    assert subprocess.run(
        ["bash", "-c", _with_truthful_build_status(synthetic)]
    ).returncode == 7

    read = "rg -n 'parallel_reduce' core/src | head -n 20"
    assert _with_truthful_build_status(read) == read


def test_managed_checkpoint_waits_for_a_stable_late_dirty_state() -> None:
    from notes.dsh_inkling.harness.contree.run_task_contree import (
        _should_run_managed_checkpoint,
    )

    state = {
        "calls": 32,
        "checkpoint_count": 0,
        "clean_diff_hash": "clean",
        "current_diff_hash": "edit-1",
        "last_checkpoint_hash": None,
        "last_successful_target_hash": None,
        "command_changed_repo": False,
        "is_build": False,
    }
    assert _should_run_managed_checkpoint(**state)
    assert not _should_run_managed_checkpoint(**(state | {"calls": 31}))
    assert not _should_run_managed_checkpoint(**(state | {"command_changed_repo": True}))
    assert not _should_run_managed_checkpoint(**(state | {"is_build": True}))
    assert not _should_run_managed_checkpoint(
        **(state | {"last_checkpoint_hash": "edit-1"})
    )
    assert not _should_run_managed_checkpoint(
        **(state | {"last_successful_target_hash": "edit-1"})
    )


def test_driver_uses_the_guarded_verifier() -> None:
    """The published payload's own test.sh awards reward 1 when zero tests run.

    Its gtest filter (`*CoreUnitTest_Serial1*`) matches no test name, so an unmodified
    repository scores a pass. `prepared_kokkos_tasks` regenerates a guarded verifier whose
    filter is real and whose coverage check fails on a zero-test selection. Using the raw
    payload inflated this project's scores by roughly 12 points per run before it was
    caught, so the driver must go through the prepared copy.
    """
    source = (HERE / "run_task_contree.py").read_text()
    assert "prepared_kokkos_tasks" in source, (
        "the driver must not grade with the raw published test.sh"
    )
    # And it must not load tasks directly from the payload.
    assert "load_harbor_tasks_from_dir" not in source


def test_target_brief_names_the_verifiers_own_targets(tmp_path: Path) -> None:
    """The brief must derive targets from the verifier, not hardcode one.

    The measured failure is that the model compiles the library while grading compiles a
    unit-test target; only 12% of building episodes built a test target. The brief is only
    trustworthy if it names what the grader actually runs, so it is read from the same
    test.sh the grader uses.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parents[4]))
    from notes.dsh_inkling.harness.contree.run_task_contree import _target_brief
    from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks

    with prepared_kokkos_tasks(
        Path(__file__).parents[4] / "data/kokkos/SWE-kokkos-bench-v2"
    ) as prepared:
        task = next(t for t in prepared if t.task_name == "kokkos__kokkos-8190")
        brief = _target_brief(task)
    assert "Kokkos_CoreUnitTest_Serial1" in brief, brief
    assert "not just the library" in brief


def test_env_brief_reports_layout_and_build_facts(tmp_path: Path) -> None:
    """The brief is read from the checkout, and must state the facts episodes rediscover."""
    from notes.dsh_inkling.harness.contree.run_task_contree import (
        _env_brief_script,
        _format_env_brief,
    )

    repo = tmp_path / "repo"
    (repo / "core" / "src" / "View").mkdir(parents=True)
    (repo / "core" / "src" / "impl").mkdir()
    (repo / "core" / "unit_test").mkdir()
    (repo / "build" / "core" / "unit_test").mkdir(parents=True)
    (repo / "build" / "CMakeCache.txt").write_text(
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "Kokkos_CXX_STANDARD:STRING=20\n"
        "Kokkos_ENABLE_SERIAL:BOOL=ON\n"
        "Kokkos_ENABLE_CUDA:BOOL=OFF\n"
    )
    (repo / "build" / "KokkosCore_config.h").write_text("")
    exe = repo / "build" / "core" / "unit_test" / "Kokkos_CoreUnitTest_Serial1"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)

    raw = subprocess.run(
        ["bash", "-c", _env_brief_script(str(repo))], capture_output=True, text=True, check=True
    ).stdout
    brief = _format_env_brief(raw)

    assert "Subdirectories of core/src: View impl" in brief
    assert "there is no core/include" in brief
    assert "already configured with Ninja" in brief
    assert "Kokkos_CXX_STANDARD=20" in brief
    assert "Kokkos_ENABLE_SERIAL=ON" in brief
    assert "CUDA" not in brief
    assert "build/core/unit_test/Kokkos_CoreUnitTest_Serial1" in brief
    assert "--gtest_filter" in brief


def test_env_brief_is_capped_and_empty_when_nothing_is_found() -> None:
    from notes.dsh_inkling.harness.contree.run_task_contree import _format_env_brief

    assert _format_env_brief("  \n") == ""
    long = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    assert len(_format_env_brief(long)) < 3100


def test_grader_timeout_is_raised_but_known_insufficient() -> None:
    """The grader timeout was raised, but measurement showed it is not the binding limit.

    A 1200s default failed kokkos__kokkos-8182 in 10 of 11 runs, finishing at 1367-1410s.
    Raising it to 2400s and re-running did NOT fix it: the episode then ran 2686s and the
    backend still reported a timeout with an empty grading log, because the sandbox's own
    3600s lifetime is what the rollout plus a full containers unit-test build exhaust.

    The default is kept higher because it is correct for the tasks whose verifier build is
    merely slow, and this test records that it does not address the pathological case, so the
    task is not silently assumed fixed.
    """
    source = (HERE / "run_task_contree.py").read_text()
    match = re.search(r'"--grader-timeout",\s*type=int,\s*default=(\d+)', source)
    assert match, "grader timeout default not found"
    assert int(match.group(1)) >= 2400, "slow verifier builds need a budget above 1200s"
    # The sandbox ceiling is what actually binds for the largest builds.
    assert "3600" in source or "sandbox_timeout" in source
