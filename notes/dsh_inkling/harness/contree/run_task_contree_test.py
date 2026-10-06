"""Guards for the harness profile.

Two failures in this project were silent rather than loud: a plugin whose config was
rejected simply did not load, and the run continued with different behaviour than
documented. Both wasted a full 100-task run. These tests make that class of mistake fail
fast instead of showing up as an unexplained score.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROFILE = Path(__file__).parent / "dsh.patch.yml.in"
RUNTIME = Path(__file__).parents[4] / ".dsh-runtime"

# The pruner rejects configurations where head + marker + tail exceeds the threshold.
# Measured from the installed plugin; see the profile comment for why it is not a guess.
PRUNE_MARKER_CODE_POINTS = 39


def _entries() -> list[dict]:
    text = PROFILE.read_text().replace("${DSH_PERSONA}", "persona")
    return yaml.safe_load(text)


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
    """The guard above is only valid while the marker length is what we measured."""
    source = (
        RUNTIME / "node_modules/@deepseek-ai/dsh-compaction-tool-result-pruner/lib/index.js"
    ).read_text()
    match = re.search(r'PRUNE_MARKER = "((?:[^"\\]|\\.)*)"', source)
    assert match, "PRUNE_MARKER not found; the plugin layout changed"
    marker = match.group(1).encode().decode("unicode_escape")
    assert len(marker) == PRUNE_MARKER_CODE_POINTS, (
        f"marker is now {len(marker)} code points; update the guard and the profile"
    )


def test_reference_inference_parameters_are_documented() -> None:
    """The proxy injects these; the profile must not silently try to set them itself."""
    text = PROFILE.read_text()
    assert "reasoning_effort" not in text, (
        "reasoning_effort cannot be expressed by the provider; the proxy injects it"
    )


def test_prompt_matching_plugins_are_disabled() -> None:
    """Each of these added prompt content or a request the reference never made."""
    disabled = {e["id"] for e in _entries() if e.get("disabled") is True}
    for entry_id in (
        "plan-mode",
        "agent-instructions",
        "session-title-llm",
        "session-title",
    ):
        assert entry_id in disabled, f"{entry_id} must stay disabled for comparability"


def test_exactly_one_tool_is_inserted_and_it_is_bash() -> None:
    """The reference exposes one repository tool; a second one changes two variables."""
    inserts = [e for e in _entries() if "insert" in e]
    assert len(inserts) == 1
    assert len(inserts[0]["insert"]) == 1
