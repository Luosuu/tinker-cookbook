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
