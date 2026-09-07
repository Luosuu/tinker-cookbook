"""Normalize framework selectors without guessing which regression should run."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.dataset import runtime_coverage
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance

FILTER = re.compile(r"(--gtest_filter(?:=|\s+))(\"[^\"]*\"|'[^']*'|[^\s;&|]+)")


def _renamed_cases(test_patch: str) -> dict[str, str]:
    # Only infer an unambiguous prefix addition evidenced by both removed and
    # added TEST declarations, e.g. space_aware_accessor -> mdspan_space_aware_accessor.
    def cases(prefix: str) -> set[str]:
        text = "\n".join(
            line[1:]
            for line in test_patch.splitlines()
            if line.startswith(prefix) and not line.startswith(prefix * 3)
        )
        return set(re.findall(r"\bTEST(?:_F|_P)?\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)\s*\)", text))

    before, after = cases("-"), cases("+")
    result = {}
    for old in before - after:
        candidates = [new for new in after - before if new.endswith("_" + old)]
        if len(candidates) == 1:
            result[old] = candidates[0]
    return result


def normalize_filter(selector: str, *, renames: dict[str, str] | None = None) -> str:
    def pattern(value: str) -> str:
        if not value:
            return value
        value = re.sub(r"\bTEST_CATEGORY(?:_DEATH)?(?=\.)", "*", value)
        suite, dot, case = value.rpartition(".")
        if not dot:
            if value.startswith("*"):
                return value
            suite, case = "*", value
        for old, new in (renames or {}).items():
            if case == old or case == old + "*":
                case = new + case[len(old) :]
                break
        return suite + "." + case

    positive, separator, negative = selector.partition("-")
    return ":".join(pattern(p) for p in positive.split(":")) + (
        "-" + ":".join(pattern(p) for p in negative.split(":")) if separator else ""
    )


def normalize_test_command(
    command: str,
    *,
    test_patch: str = "",
    after_test_patch: bool = True,
    instance: KokkosInstance | None = None,
) -> str:
    renames = _renamed_cases(test_patch) if after_test_patch else {}
    reviewed: dict[str, str] = {}
    if instance is not None:
        overrides = json.loads(Path(__file__).with_name("test_command_overrides.json").read_text())
        entry = overrides.get(instance.instance_id)
        if entry is not None and entry["base_commit"] == instance.base_commit:
            reviewed = entry.get("selectors", {})
            command = entry.get("commands", {}).get(command, command)
    # Quoting the whole --flag=value argument is valid shell syntax too.
    command = re.sub(
        r"(['\"])(--gtest_filter=)([^'\"]*)\1",
        lambda match: match.group(2) + shlex.quote(match.group(3)),
        command,
    )

    def replace_filter(match: re.Match[str]) -> str:
        selector = shlex.split(match.group(2))
        if len(selector) != 1:
            raise ValueError("Ambiguous GoogleTest filter")
        value = selector[0]
        # Some historical annotations escaped their quote delimiters, passing
        # literal quotes into GoogleTest rather than using shell quoting.
        while len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
            value = value[1:-1]
        if "'" in value or '"' in value:
            raise ValueError("GoogleTest filter contains an ambiguous literal quote")
        value = reviewed.get(value, value)
        return "--gtest_filter=" + shlex.quote(normalize_filter(value, renames=renames))

    command = FILTER.sub(replace_filter, command)
    ctest = re.search(r"(?:^|[;&|]\s*)ctest(?=\s|$)", command)
    if (
        ctest
        and not re.search(r"--test-dir(?:[=\s]|$)", command)
        and not re.search(r"\bcd\s", command[: ctest.end()])
    ):
        command = re.sub(r"\bctest(?=\s|$)", "ctest --test-dir build", command, count=1)
    if ctest and not re.search(r"(?:^|\s)(?:-V|--verbose)(?:\s|$)", command):
        # Inspect nested GoogleTest output too; CTest can pass one executable
        # whose own filter selected zero tests. Works with older bundled CTest.
        command = command.replace("ctest", "ctest --verbose", 1)
    return command


def guard_source() -> str:
    return Path(runtime_coverage.__file__).read_text()


def guarded_command(command: str) -> str:
    return "python3 -c " + shlex.quote(guard_source()) + " " + shlex.quote(command)
