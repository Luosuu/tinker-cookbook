"""Tests for the write-detection heuristic.

The analysis conclusions depend on distinguishing "the agent edited the
repository" from "the agent read the repository with a shell redirect". These
cases pin that boundary, because a false positive invents agents that acted when
they only looked, and a false negative does the reverse.
"""

from __future__ import annotations

import pytest

from notes.dsh_inkling.analyze_trajectories import _is_write


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn 'to_array' /workspace/repo/core/src 2>/dev/null | head -n 40",
        "find /workspace/repo -name '*.hpp' 2>/dev/null | head -n 5",
        "ls /workspace/repo 2>/dev/null",
        "sed -n '160,240p' /workspace/repo/core/src/impl/Kokkos_ChaseLev.hpp",
        "cat /workspace/repo/core/src/Kokkos_Atomic.hpp",
        "cmake --build build --parallel 4 > /dev/null 2>&1",
    ],
)
def test_reads_are_not_writes(command: str) -> None:
    assert not _is_write(command)


@pytest.mark.parametrize(
    "command",
    [
        "cat > /tmp/test_traits.cpp << 'EOF'\nint main() {}\nEOF",
        "sed -i 's/atomic_increment/atomic_inc/' core/src/Kokkos_Crs.hpp",
        "echo 'int x;' >> /workspace/repo/core/src/new.hpp",
        "git apply /tmp/fix.patch",
        "patch -p1 < /tmp/fix.diff",
        "cat core/src/Kokkos_Crs.hpp > /workspace/repo/core/src/Kokkos_Crs.hpp.bak",
    ],
)
def test_edits_are_writes(command: str) -> None:
    assert _is_write(command)


def test_empty_command_is_not_a_write() -> None:
    assert not _is_write("")
