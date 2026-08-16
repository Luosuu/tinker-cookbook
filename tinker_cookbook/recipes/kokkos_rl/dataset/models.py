"""Serializable types for Kokkos coding-RL candidates and validated instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str
    additions: int
    deletions: int

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @classmethod
    def from_api(cls, value: dict[str, Any]) -> ChangedFile:
        return cls(
            filename=str(value["filename"]),
            status=str(value.get("status", "modified")),
            additions=int(value.get("additions", 0)),
            deletions=int(value.get("deletions", 0)),
        )


@dataclass(frozen=True)
class KokkosInstance:
    """A raw candidate becomes validated once its verification fields are populated."""

    instance_id: str
    repo: str
    pr_number: int
    title: str
    problem_statement: str
    merged_at: str
    base_commit: str
    merge_commit: str
    era: str
    labels: tuple[str, ...]
    changed_files: tuple[ChangedFile, ...]
    patch: str
    test_patch: str
    code_patch: str
    linked_issues: tuple[int, ...] = ()
    configure_command: str = (
        "cmake -S . -B build -G Ninja "
        "-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON "
        "-DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF "
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    )
    build_targets: tuple[str, ...] = ()
    build_command: str = ""
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()
    f2p_commands: tuple[str, ...] = ()
    p2p_commands: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_validation_ready(self) -> bool:
        has_f2p = bool(self.f2p_commands) or self.metadata.get("f2p_stage") == "build"
        has_build = bool(self.build_targets or self.build_command)
        return bool(has_build and has_f2p and self.test_patch and self.code_patch)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve standard SWE-bench names for downstream compatibility.
        value["FAIL_TO_PASS"] = list(self.fail_to_pass)
        value["PASS_TO_PASS"] = list(self.pass_to_pass)
        value["hints_text"] = ""
        value["created_at"] = self.merged_at
        value["version"] = self.era
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KokkosInstance:
        def tuple_of_strings(key: str, fallback: str | None = None) -> tuple[str, ...]:
            raw = value.get(key, value.get(fallback, [])) if fallback else value.get(key, [])
            return tuple(str(item) for item in raw)

        return cls(
            instance_id=str(value["instance_id"]),
            repo=str(value.get("repo", "kokkos/kokkos")),
            pr_number=int(value["pr_number"]),
            title=str(value["title"]),
            problem_statement=str(value["problem_statement"]),
            merged_at=str(value["merged_at"]),
            base_commit=str(value["base_commit"]),
            merge_commit=str(value["merge_commit"]),
            era=str(value["era"]),
            labels=tuple_of_strings("labels"),
            changed_files=tuple(ChangedFile(**item) for item in value.get("changed_files", [])),
            patch=str(value.get("patch", "")),
            test_patch=str(value.get("test_patch", "")),
            code_patch=str(value.get("code_patch", "")),
            linked_issues=tuple(int(item) for item in value.get("linked_issues", [])),
            configure_command=str(value.get("configure_command", cls.configure_command)),
            build_targets=tuple_of_strings("build_targets"),
            build_command=str(value.get("build_command", "")),
            fail_to_pass=tuple_of_strings("fail_to_pass", "FAIL_TO_PASS"),
            pass_to_pass=tuple_of_strings("pass_to_pass", "PASS_TO_PASS"),
            f2p_commands=tuple_of_strings("f2p_commands"),
            p2p_commands=tuple_of_strings("p2p_commands"),
            metadata=dict(value.get("metadata", {})),
        )
