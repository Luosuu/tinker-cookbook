"""Kokkos-specific patch classification and candidate filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise

from tinker_cookbook.recipes.kokkos_rl.models import ChangedFile

SOURCE_PREFIXES = ("core/src/", "containers/src/", "algorithms/src/")
GPU_BACKEND_SEGMENTS = frozenset({"Cuda", "CUDA", "HIP", "SYCL", "OpenMPTarget", "OpenACC"})
UNSUPPORTED_BACKEND_SEGMENTS = GPU_BACKEND_SEGMENTS | {"HPX"}
SUPPORTED_HOST_BACKEND_SEGMENTS = frozenset({"Serial", "OpenMP", "Threads"})
ISSUE_REFERENCE_RE = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?[ \t]+#(\d+)"
)
ISSUE_URL_RE = re.compile(r"https?://github\.com/kokkos/kokkos/issues/(\d+)")
GPU_TITLE_RE = re.compile(r"(?i)\b(?:cuda|hip|sycl|rocm|nvidia|amd gpu|rubin)\b")
GPU_COMPILER_RE = re.compile(r"(?i)\b(?:nvcc|hipcc)\b")
MAINTENANCE_TITLE_RE = re.compile(
    r"(?i)\b(?:deprecat(?:e|ed|ion)|final removal|remove deprecated|cleanup only)\b"
)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
SOURCE_LINE_URL_RE = re.compile(
    r"https?://github\.com/[^\s]+/blob/[^\s#]+(?:/[^\s#]+)*#L\d+(?:-L\d+)?"
)


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    return any(part in {"unit_test", "unit_tests"} for part in parts)


def is_source_path(path: str) -> bool:
    return path.startswith(SOURCE_PREFIXES)


def is_gpu_backend_path(path: str) -> bool:
    return bool(GPU_BACKEND_SEGMENTS.intersection(path.split("/")))


def is_forbidden_agent_path(path: str) -> bool:
    """Paths an agent may not change because they can compromise scoring."""

    return (
        is_test_path(path)
        or path.startswith((".github/", "cmake/"))
        or path.endswith("CMakeLists.txt")
    )


def linked_issue_numbers(text: str) -> tuple[int, ...]:
    matches = ISSUE_REFERENCE_RE.findall(text) + ISSUE_URL_RE.findall(text)
    return tuple(dict.fromkeys(int(match) for match in matches))


def sanitize_problem_statement(text: str) -> str:
    """Remove common gold-patch leakage while retaining the user-facing bug report."""

    text = FENCED_CODE_RE.sub("[implementation suggestion omitted]", text)
    text = SOURCE_LINE_URL_RE.sub("[source location omitted]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_unified_diff(patch: str) -> list[tuple[str, str]]:
    """Return ``(new_path, diff_block)`` pairs from a git unified diff."""

    starts = [match.start() for match in re.finditer(r"(?m)^diff --git ", patch)]
    if not starts:
        return []
    starts.append(len(patch))
    blocks: list[tuple[str, str]] = []
    for start, end in pairwise(starts):
        block = patch[start:end]
        path_match = re.search(r"(?m)^\+\+\+ (?:b/)?([^\t\n]+)", block)
        if path_match is None or path_match.group(1) == "/dev/null":
            header = re.match(r"diff --git a/(.+?) b/(.+)\n", block)
            path = header.group(2) if header else ""
        else:
            path = path_match.group(1)
        blocks.append((path, block))
    return blocks


def partition_patch(patch: str) -> tuple[str, str]:
    """Split a PR patch into hidden tests and agent-visible production changes."""

    test_blocks: list[str] = []
    code_blocks: list[str] = []
    for path, block in split_unified_diff(patch):
        (test_blocks if is_test_path(path) else code_blocks).append(block)
    return "".join(test_blocks), "".join(code_blocks)


def candidate_rejection_reasons(
    files: Iterable[ChangedFile],
    *,
    title: str = "",
    description: str = "",
    min_changed_lines: int = 5,
    max_changed_lines: int = 500,
) -> tuple[str, ...]:
    files = tuple(files)
    source_files = tuple(item for item in files if is_source_path(item.filename))
    test_files = tuple(item for item in files if is_test_path(item.filename))
    changed_lines = sum(item.changed_lines for item in files)
    reasons: list[str] = []
    if not source_files:
        reasons.append("no-production-source-change")
    if not test_files:
        reasons.append("no-unit-test-change")
    if source_files and all(is_gpu_backend_path(item.filename) for item in source_files):
        reasons.append("gpu-backend-only")
    elif source_files and all(
        UNSUPPORTED_BACKEND_SEGMENTS.intersection(item.filename.split("/"))
        and not SUPPORTED_HOST_BACKEND_SEGMENTS.intersection(item.filename.split("/"))
        for item in source_files
    ):
        reasons.append("unsupported-backend-only")
    elif GPU_TITLE_RE.search(title):
        reasons.append("gpu-specific-change")
    elif GPU_COMPILER_RE.search(description):
        reasons.append("gpu-compiler-specific-change")
    if MAINTENANCE_TITLE_RE.search(title):
        reasons.append("maintenance-cleanup")
    if changed_lines < min_changed_lines:
        reasons.append("diff-too-small")
    if changed_lines > max_changed_lines:
        reasons.append("diff-too-large")
    return tuple(reasons)


def infer_era(merged_at: str) -> str:
    """Choose the construction image era; validation remains authoritative."""

    date = merged_at[:10]
    if date >= "2025-12-01":
        return "cpp20"
    if date >= "2023-01-01":
        return "cpp17"
    return "cpp14"
