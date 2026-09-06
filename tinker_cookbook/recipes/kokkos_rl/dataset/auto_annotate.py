"""Generate Kokkos verification annotations with an LLM and verify them on Modal."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shlex
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

import aiohttp
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.kokkos_rl.chat_inference import (
    TINKER_CHAT_BASE_URL,
    ChatAnnotationCompleter,
    prepare_annotation_state,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.annotate import apply_annotations
from tinker_cookbook.recipes.kokkos_rl.dataset.ecosystem import get_repository_profile
from tinker_cookbook.recipes.kokkos_rl.dataset.modal_validate import (
    ModalValidationSandboxFactory,
    contree_validation_sandbox_factory,
    default_validation_sandbox_factory,
    validate_instance_in_sandbox,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.dataset.test_commands import (
    normalize_filter,
    normalize_test_command,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_OUTPUT_FIELDS = {
    "instance_id",
    "configure_command",
    "build_command",
    "build_targets",
    "fail_to_pass",
    "pass_to_pass",
    "f2p_commands",
    "p2p_commands",
    "metadata",
}

SYSTEM_PROMPT = """You construct executable verification metadata for Kokkos bug-fix PRs.
Return exactly one JSON object and no prose. Never copy implementation details into metadata.
The JSON schema is:
{
  "instance_id": "same id as input",
  "configure_command": "cmake configure command",
  "build_command": "exact full build command, or empty when build_targets are used",
  "build_targets": ["exact CMake target"],
  "fail_to_pass": ["human-readable new test id or compile target"],
  "pass_to_pass": ["stable existing test id"],
  "f2p_commands": ["exact command, run from /workspace/repo"],
  "p2p_commands": ["exact command, run from /workspace/repo"],
  "metadata": {"f2p_stage": "build or test", "toolchain": "cpu", "requires_gpu": false}
}

Use f2p_stage=build when applying the test patch to the parent should make the selected target
fail to compile; otherwise use f2p_stage=test and provide at least one f2p_commands entry. Select
the narrowest targets that compile the added tests. Prefer exact gtest filters and anchored ctest
regexes. CTest -R matches registered executable test names, not gtest suite/case names; invoke the
built executable with --gtest_filter for suite/case selection. P2P checks must already pass on the
parent. Do not invent files or targets that are not
supported by the diff. For CMake repositories, build_targets are logical CMake/Ninja targets,
never source basenames or filesystem paths. Kokkos core unit-test translation units used only for
static assertions may be registered under the aggregate Kokkos_CoreTestCompileOnly target. In a
Kokkos core CMake list, TESTNAMES/SOURCES1 feeds Kokkos_CoreUnitTest_Serial1 and
TESTNAMES/SOURCES2 feeds Kokkos_CoreUnitTest_Serial2. Treat supplied build-context snippets as
authoritative when selecting among those targets. For a
Python repository, leave build_targets empty and provide a narrow build_command such as compileall;
use pytest selectors for F2P/P2P. Set requires_gpu=false for compilation, static assertions, and
tests that do not execute accelerator code. Set it true only when the final passing verification
must execute kernels or inspect a real GPU device. Choose toolchain independently from GPU access:
cpu, cuda, hip, or sycl. A generic production fix may use the CPU toolchain even when the PR also
adds backend coverage; a compiler-specific regression needs its matching toolchain. Do not use
network access in verification commands."""


class AnnotationCompleter(Protocol):
    async def __call__(self, messages: list[Message]) -> Message: ...


class TinkerAnnotationCompleter:
    """Tinker message completer with explicit Inkling effort conditioning."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: Renderer,
        *,
        max_tokens: int,
        temperature: float,
        thinking_effort: float | None,
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        if thinking_effort is not None and not (
            math.isfinite(thinking_effort) and 0.0 <= thinking_effort < 1.0
        ):
            raise ValueError("thinking_effort must be finite and in [0.0, 1.0)")
        self.thinking_effort = thinking_effort

    async def __call__(self, messages: list[Message]) -> Message:
        if self.thinking_effort is None:
            model_input = self.renderer.build_generation_prompt(messages)
        elif isinstance(self.renderer, TmlV0Renderer):
            model_input = self.renderer.build_generation_prompt(
                messages, effort=self.thinking_effort
            )
        else:
            raise ValueError("thinking_effort is only supported by the Inkling tml_v0 renderer")
        response = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=self.renderer.get_stop_sequences(),
            ),
        )
        parsed_message, _termination = self.renderer.parse_response(response.sequences[0].tokens)
        return parsed_message


ANNOTATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "instance_id": {"type": "string"},
        "configure_command": {"type": "string"},
        "build_command": {"type": "string"},
        "build_targets": {"type": "array", "items": {"type": "string"}},
        "fail_to_pass": {"type": "array", "items": {"type": "string"}},
        "pass_to_pass": {"type": "array", "items": {"type": "string"}},
        "f2p_commands": {"type": "array", "items": {"type": "string"}},
        "p2p_commands": {"type": "array", "items": {"type": "string"}},
        "metadata": {
            "type": "object",
            "properties": {
                "f2p_stage": {"type": "string", "enum": ["build", "test"]},
                "toolchain": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "hip", "sycl"],
                },
                "requires_gpu": {"type": "boolean"},
            },
            "required": ["f2p_stage", "toolchain", "requires_gpu"],
            "additionalProperties": False,
        },
    },
    "required": [
        "instance_id",
        "configure_command",
        "build_command",
        "build_targets",
        "fail_to_pass",
        "pass_to_pass",
        "f2p_commands",
        "p2p_commands",
        "metadata",
    ],
    "additionalProperties": False,
}


class OpenAIAnnotationCompleter:
    """Responses API completer using strict Structured Outputs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int,
        reasoning_effort: str,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI annotation provider")
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    async def __call__(self, messages: list[Message]) -> Message:
        input_messages = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("OpenAI annotation prompts must contain text-only messages")
            input_messages.append({"role": message["role"], "content": content})
        payload = {
            "model": self.model,
            "input": input_messages,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "kokkos_verification_annotation",
                    "strict": True,
                    "schema": ANNOTATION_JSON_SCHEMA,
                }
            },
        }
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                value = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"OpenAI Responses API returned HTTP {response.status}: {str(value)[:2000]}"
                    )
        output_texts = [
            content.get("text", "")
            for item in value.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        if not output_texts:
            raise RuntimeError(f"OpenAI response contained no output text: {str(value)[:2000]}")
        return {"role": "assistant", "content": "".join(output_texts)}


@dataclass
class AnnotationAttempt:
    attempt: int
    raw_output: str = ""
    annotation: dict[str, object] | None = None
    validation: dict[str, object] | None = None
    error: str = ""


@dataclass
class AutoAnnotationResult:
    instance_id: str
    passed: bool = False
    validated_instance: KokkosInstance | None = None
    attempts: list[AnnotationAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "validated_instance": (
                self.validated_instance.to_dict() if self.validated_instance else None
            ),
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> AutoAnnotationResult:
        raw_instance = value.get("validated_instance")
        raw_attempts = value.get("attempts", [])
        if not isinstance(raw_attempts, list):
            raise ValueError("cached attempts must be an array")
        return cls(
            instance_id=str(value["instance_id"]),
            passed=bool(value.get("passed", False)),
            validated_instance=(
                KokkosInstance.from_dict(raw_instance) if isinstance(raw_instance, dict) else None
            ),
            attempts=[
                AnnotationAttempt(**attempt)
                for attempt in raw_attempts
                if isinstance(attempt, dict)
            ],
        )


def _candidate_payload(instance: KokkosInstance) -> dict[str, object]:
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "title": instance.title,
        "problem_statement": instance.problem_statement,
        "era": instance.era,
        "changed_files": [asdict(item) for item in instance.changed_files],
        "default_configure_command": instance.configure_command,
        "candidate_metadata": instance.metadata,
        "test_patch": instance.test_patch,
        "code_patch": instance.code_patch,
    }


def _candidate_cmake_paths(instance: KokkosInstance) -> list[str]:
    paths: set[str] = set()
    for item in instance.changed_files:
        if "test" not in item.filename.lower():
            continue
        parent = PurePosixPath(item.filename).parent
        while str(parent) not in {"", "."}:
            paths.add(str(parent / "CMakeLists.txt"))
            parent = parent.parent
    return sorted(paths, key=lambda value: (value.count("/"), value), reverse=True)[:3]


def _fetch_cmake_context_sync(instance: KokkosInstance) -> str:
    stems = {
        PurePosixPath(item.filename).stem.removeprefix("Test")
        for item in instance.changed_files
        if "test" in item.filename.lower()
    }
    sections: list[str] = []
    for path in _candidate_cmake_paths(instance):
        url = f"https://raw.githubusercontent.com/{instance.repo}/{instance.base_commit}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode(errors="replace")
        except (OSError, urllib.error.HTTPError, urllib.error.URLError):
            continue
        lines = text.splitlines()
        selected: set[int] = set()
        for index, line in enumerate(lines):
            if any(stem and stem in line for stem in stems):
                selected.update(range(max(0, index - 90), min(len(lines), index + 25)))
            if "kokkos_add_executable" in line or "add_executable" in line:
                selected.update(range(max(0, index - 3), min(len(lines), index + 4)))
        if selected:
            rendered = "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(selected))
            sections.append(f"# {path}\n{rendered}")
    return "\n\n".join(sections)[:30000]


async def _fetch_cmake_context(instance: KokkosInstance) -> str:
    if get_repository_profile(instance.repo).build_system != "cmake":
        return ""
    return await asyncio.to_thread(_fetch_cmake_context_sync, instance)


def _infer_core_target_from_context(instance: KokkosInstance, build_context: str) -> str:
    markers = {
        "Kokkos_CoreTestCompileOnly": ("COMPILE_ONLY_SOURCES",),
        "Kokkos_CoreUnitTest_Serial1": ("TESTNAMES1", "SOURCES1"),
        "Kokkos_CoreUnitTest_Serial2": ("TESTNAMES2", "SOURCES2"),
    }
    for item in instance.changed_files:
        if not item.filename.startswith("core/unit_test/"):
            continue
        stem = PurePosixPath(item.filename).stem.removeprefix("Test")
        match = re.search(rf"(?m)^\d+:.*\b{re.escape(stem)}\b", build_context)
        if match is None:
            continue
        preceding = build_context[: match.start()]
        ranked = [
            (preceding.rfind(marker), target)
            for target, target_markers in markers.items()
            for marker in target_markers
        ]
        position, target = max(ranked)
        if position >= 0:
            return target
    return ""


def _prompt(
    instance: KokkosInstance,
    *,
    previous_annotation: dict[str, object] | None,
    feedback: str,
    build_context: str = "",
) -> list[Message]:
    body = "Candidate:\n" + json.dumps(_candidate_payload(instance), indent=2)
    if build_context:
        body += "\n\nBuild context from the candidate base commit:\n" + build_context
    if previous_annotation is not None:
        body += "\n\nPrevious annotation:\n" + json.dumps(previous_annotation, indent=2)
    if feedback:
        body += (
            "\n\nThe previous proposal failed real sandbox validation. Correct the annotation "
            "using this evidence:\n" + feedback[-24000:]
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]


def parse_annotation(text: str, instance_id: str) -> dict[str, object]:
    """Parse and strictly validate a model-produced annotation object."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model output does not contain a JSON object") from None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid annotation JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("annotation must be a JSON object")

    unknown = set(value) - _OUTPUT_FIELDS
    if unknown:
        raise ValueError(f"unsupported annotation fields: {sorted(unknown)}")
    output_instance_id = value.pop("instance_id", instance_id)
    if output_instance_id != instance_id:
        raise ValueError(
            f"annotation instance_id {output_instance_id!r} does not match {instance_id!r}"
        )

    for key in (
        "build_targets",
        "fail_to_pass",
        "pass_to_pass",
        "f2p_commands",
        "p2p_commands",
    ):
        items = value.setdefault(key, [])
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError(f"{key} must be an array of non-empty strings")
        if any("\n" in item or "\0" in item for item in items):
            raise ValueError(f"{key} entries must be single-line strings")

    targets = value["build_targets"]
    assert isinstance(targets, list)
    if any(not _TARGET_RE.fullmatch(target) for target in targets):
        raise ValueError("build_targets must contain valid CMake target names")

    build_command = value.setdefault("build_command", "")
    if not isinstance(build_command, str):
        raise ValueError("build_command must be a string")
    if "\n" in build_command or "\0" in build_command:
        raise ValueError("build_command must be a single-line string")
    if not targets and not build_command.strip():
        raise ValueError("provide at least one build target or a build_command")

    configure_command = value.get("configure_command")
    if configure_command is not None:
        if not isinstance(configure_command, str) or not configure_command.strip():
            raise ValueError("configure_command must be a non-empty string")
        if "\n" in configure_command or "\0" in configure_command:
            raise ValueError("configure_command must be a single-line string")

    metadata = value.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    unknown_metadata = set(metadata) - {"f2p_stage", "requires_gpu", "toolchain"}
    if unknown_metadata:
        raise ValueError(f"unsupported metadata fields: {sorted(unknown_metadata)}")
    stage = metadata.setdefault("f2p_stage", "test")
    if stage not in {"build", "test"}:
        raise ValueError("metadata.f2p_stage must be 'build' or 'test'")
    if stage == "test" and not value["f2p_commands"]:
        raise ValueError("test-stage annotations require at least one f2p command")
    requires_gpu = metadata.setdefault("requires_gpu", False)
    if not isinstance(requires_gpu, bool):
        raise ValueError("metadata.requires_gpu must be a boolean")
    toolchain = metadata.setdefault("toolchain", "cpu")
    if toolchain not in {"cpu", "cuda", "hip", "sycl"}:
        raise ValueError("metadata.toolchain must be cpu, cuda, hip, or sycl")
    if requires_gpu and toolchain != "cuda":
        raise ValueError("Modal GPU execution currently requires the cuda toolchain")
    return value


def _validation_feedback(validation: dict[str, object]) -> str:
    results = validation.get("results", [])
    if not isinstance(results, list):
        results = []
    compact_results = []
    for item in results[-3:]:
        if isinstance(item, dict):
            compact_results.append(
                {
                    "command": item.get("command"),
                    "expected_exit": item.get("expected_exit"),
                    "exit_code": item.get("exit_code"),
                    "stdout": str(item.get("stdout", ""))[-6000:],
                    "stderr": str(item.get("stderr", ""))[-6000:],
                }
            )
    return json.dumps(
        {"error": validation.get("error", ""), "last_results": compact_results},
        indent=2,
    )


def _normalize_toolchain_annotation(
    annotation: dict[str, object],
    instance: KokkosInstance,
    *,
    attempt_number: int = 1,
    fallback_attempt: int = 3,
    build_context: str = "",
) -> dict[str, object]:
    """Make mechanical compiler settings deterministic after the model chooses a toolchain."""

    profile = get_repository_profile(instance.repo)
    metadata = annotation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    test_paths = [
        item.filename for item in instance.changed_files if "test" in item.filename.lower()
    ]
    uses_recent_core_unit_tests = (
        instance.repo == "kokkos/kokkos"
        and metadata.get("toolchain") == "cpu"
        and instance.merged_at[:10] >= "2024-01-01"
        and any(path.startswith("core/unit_test/") for path in test_paths)
    )
    inferred_core_target = (
        _infer_core_target_from_context(instance, build_context)
        if uses_recent_core_unit_tests
        else ""
    )
    targets = annotation.get("build_targets")
    if instance.repo == "kokkos/kokkos" and isinstance(targets, list):
        generic_targets = {"all", "clean", "install", "test", "kokkoscore"}
        annotation["build_targets"] = [
            (
                target
                if not isinstance(target, str)
                or target.startswith("Kokkos_")
                or target in generic_targets
                else f"Kokkos_{target}"
            )
            for target in targets
        ]
        targets = annotation["build_targets"]
    if inferred_core_target:
        annotation["build_command"] = ""
        annotation["build_targets"] = [inferred_core_target]
        targets = annotation["build_targets"]
    if "EXPECT_DEATH" in instance.test_patch:
        configure = annotation.get("configure_command")
        if isinstance(configure, str):
            if "-DCMAKE_BUILD_TYPE=" in configure:
                configure = re.sub(r"-DCMAKE_BUILD_TYPE=\S+", "-DCMAKE_BUILD_TYPE=Debug", configure)
            else:
                configure += " -DCMAKE_BUILD_TYPE=Debug"
            annotation["configure_command"] = configure

    if (
        instance.repo == "kokkos/kokkos"
        and metadata.get("f2p_stage") == "test"
        and isinstance(targets, list)
    ):
        for command_key in ("f2p_commands", "p2p_commands"):
            commands = annotation.get(command_key)
            if isinstance(commands, list):
                annotation[command_key] = [
                    command
                    for command in commands
                    if not isinstance(command, str)
                    or not command.lstrip().startswith("cmake --build ")
                ]
        runtime_targets = [
            target
            for target in targets
            if isinstance(target, str)
            and ("UnitTest" in target or "UnitTests" in target)
            and "CompileOnly" not in target
        ]
        if runtime_targets:
            target = runtime_targets[0]
            binary = (
                '"$(find build -type f -name ' + shlex.quote(target) + ' -perm -111 -print -quit)"'
            )
            for command_key, label_key in (
                ("f2p_commands", "fail_to_pass"),
                ("p2p_commands", "pass_to_pass"),
            ):
                commands = annotation.get(command_key)
                labels = annotation.get(label_key)
                if not isinstance(commands, list):
                    continue
                if not isinstance(labels, list):
                    labels = []
                normalized_commands: list[object] = []
                for index, command in enumerate(commands):
                    if not isinstance(command, str):
                        normalized_commands.append(command)
                        continue
                    if "--gtest_filter" in command:
                        selector = command.split("--gtest_filter", maxsplit=1)[1].lstrip(" =")
                        unquoted_selector = selector.strip("'\"")
                        selector = shlex.quote(normalize_filter(unquoted_selector))
                        normalized_commands.append(f"{binary} --gtest_filter={selector}")
                        continue
                    if not command.lstrip().startswith("ctest "):
                        normalized_commands.append(command)
                        continue
                    parts = shlex.split(command)
                    regex = parts[parts.index("-R") + 1] if "-R" in parts else ""
                    if any(target in regex for target in runtime_targets):
                        normalized_commands.append(command)
                        continue
                    selector_source = labels[min(index, len(labels) - 1)] if labels else regex
                    selector = str(selector_source).split("/")[-1]
                    selector = selector.strip(" ^$'\"")
                    normalized_commands.append(
                        f"{binary} --gtest_filter={shlex.quote(f'*{selector}*')}"
                    )
                annotation[command_key] = normalized_commands
    if (
        uses_recent_core_unit_tests
        and not inferred_core_target
        and attempt_number >= fallback_attempt
    ):
        # These headers are expanded into generated backend sources by unchanged
        # CMake. Building all three CPU aggregates is deterministic and still much
        # narrower than a full Kokkos build; one of them will own the changed test.
        annotation["build_command"] = ""
        annotation["build_targets"] = [
            "Kokkos_CoreTestCompileOnly",
            "Kokkos_CoreUnitTest_Serial1",
            "Kokkos_CoreUnitTest_Serial2",
        ]
        # Preserve requested runtime P2P checks even for compile-failure tasks.
        # A bad selector must be repaired, not silently removed to pass validation.

    if profile.full_build_for_validation:
        annotation["build_command"] = "cmake --build build --parallel"
        annotation["build_targets"] = []
    if profile.build_system == "python":
        configure = annotation.get("configure_command")
        if isinstance(configure, str) and "pip install" in configure:
            annotation["configure_command"] = configure.replace(
                "pip install", "pip install --break-system-packages", 1
            ).replace(
                "--break-system-packages --break-system-packages",
                "--break-system-packages",
            )
    if instance.repo == "kokkos/kokkos-remote-spaces":
        configure = annotation.get("configure_command")
        if isinstance(configure, str):
            configure = re.sub(r"\s+-DKokkosRemoteSpaces_REMOTE_SPACES=\S+", "", configure)
            configure = configure.replace("-DBUILD_TESTING=ON", "")
            if "-DKRS_ENABLE_MPISPACE=" not in configure:
                configure += " -DKRS_ENABLE_MPISPACE=ON"
            if "-DKRS_ENABLE_TESTS=" not in configure:
                configure += " -DKRS_ENABLE_TESTS=ON"
            annotation["configure_command"] = configure

    for key in ("f2p_commands", "p2p_commands"):
        commands = annotation.get(key)
        if isinstance(commands, list):
            annotation[key] = [
                normalize_test_command(command, after_test_patch=False)
                if isinstance(command, str)
                else command
                for command in commands
            ]

    if not isinstance(metadata, dict) or metadata.get("toolchain") != "cuda":
        return annotation
    configure = annotation.get("configure_command")
    if not isinstance(configure, str) or not configure.lstrip().startswith("cmake "):
        return annotation

    wrapper = (
        "/workspace/repo/bin/nvcc_wrapper"
        if instance.repo == "kokkos/kokkos"
        else "/workspace/kokkos/bin/nvcc_wrapper"
    )
    configure = re.sub(r"\s+-DCMAKE_CXX_COMPILER_LAUNCHER=\S+", "", configure)
    configure = re.sub(
        r"-DCMAKE_CXX_COMPILER=(?:\S*/)?nvcc(?:_wrapper)?",
        f"-DCMAKE_CXX_COMPILER={wrapper}",
        configure,
    )
    if "-DCMAKE_CXX_COMPILER=" not in configure:
        configure += f" -DCMAKE_CXX_COMPILER={wrapper}"
    configure = configure.replace("-DKokkos_ENABLE_OPENMP=ON", "-DKokkos_ENABLE_OPENMP=OFF")
    if "-DKokkos_ENABLE_CUDA=" not in configure:
        configure += " -DKokkos_ENABLE_CUDA=ON"
    if "-DKokkos_ARCH_" not in configure:
        configure += " -DKokkos_ARCH_ADA89=ON"
    annotation["configure_command"] = configure
    return annotation


async def annotate_and_validate_instance(
    instance: KokkosInstance,
    *,
    completer: AnnotationCompleter,
    max_attempts: int = 3,
    sandbox_timeout: int = 3600,
    command_timeout: int = 1200,
    flaky_repetitions: int = 3,
    sandbox_factory: ModalValidationSandboxFactory = default_validation_sandbox_factory,
) -> AutoAnnotationResult:
    """Iteratively propose annotations and retain only a Modal-verified instance."""

    result = AutoAnnotationResult(instance_id=instance.instance_id)
    previous_annotation: dict[str, object] | None = None
    feedback = ""
    build_context = await _fetch_cmake_context(instance)
    for attempt_number in range(1, max_attempts + 1):
        attempt = AnnotationAttempt(attempt=attempt_number)
        result.attempts.append(attempt)
        try:
            message = await completer(
                _prompt(
                    instance,
                    previous_annotation=previous_annotation,
                    feedback=feedback,
                    build_context=build_context,
                )
            )
            attempt.raw_output = get_text_content(message)
            annotation = parse_annotation(attempt.raw_output, instance.instance_id)
            annotation = _normalize_toolchain_annotation(
                annotation,
                instance,
                attempt_number=attempt_number,
                fallback_attempt=max_attempts,
                build_context=build_context,
            )
            attempt.annotation = annotation
            previous_annotation = annotation
            annotated = apply_annotations(instance, annotation)
            validation_report = await validate_instance_in_sandbox(
                annotated,
                sandbox_timeout=sandbox_timeout,
                command_timeout=command_timeout,
                flaky_repetitions=flaky_repetitions,
                sandbox_factory=sandbox_factory,
            )
            attempt.validation = validation_report.to_dict()
            if validation_report.passed:
                result.passed = True
                result.validated_instance = annotated
                break
            feedback = _validation_feedback(attempt.validation)
        except Exception as error:
            attempt.error = f"{type(error).__name__}: {error}"
            feedback = attempt.error
    return result


def _read_instances(
    path: Path,
    max_candidates: int | None,
    instance_id: str | None,
    offset: int = 0,
) -> list[KokkosInstance]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    instances = [
        KokkosInstance.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if instance_id is not None:
        instances = [item for item in instances if item.instance_id == instance_id]
        if not instances:
            raise ValueError(f"instance_id {instance_id!r} was not found in {path}")
    return instances[offset:][:max_candidates]


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing the process environment."""

    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--failed-output", type=Path)
    parser.add_argument("--provider", choices=("openai", "tinker", "tinker-chat"), default="openai")
    parser.add_argument("--model-name", default="thinkingmachines/Inkling:peft:262144")
    parser.add_argument("--openai-model", default="gpt-5.6-terra")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--checkpoint-url")
    parser.add_argument("--renderer-name")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking-effort", type=float, default=0.7)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--instance-id")
    parser.add_argument("--sandbox-timeout", type=int, default=3600)
    parser.add_argument("--command-timeout", type=int, default=1200)
    parser.add_argument("--flaky-repetitions", type=int, default=3)
    parser.add_argument(
        "--sandbox-backend",
        choices=("modal", "contree"),
        default="contree",
        help="cloud sandbox used to validate generated tasks (default: contree)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore all per-candidate reports from an earlier invocation",
    )
    parser.add_argument(
        "--skip-failed-reports",
        action="store_true",
        help="when resuming, do not retry candidates with an existing failed report",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    completer: AnnotationCompleter
    if args.provider == "tinker-chat" or (args.reports_dir / "inference_config.json").exists():
        prepare_annotation_state(
            args.reports_dir,
            {
                "provider": args.provider,
                "model": args.checkpoint_url or args.model_name,
                "openai_model": args.openai_model,
                "base_url": args.base_url or TINKER_CHAT_BASE_URL,
                "thinking_effort": args.thinking_effort,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_attempts": args.max_attempts,
                "sandbox_backend": args.sandbox_backend,
                "flaky_repetitions": args.flaky_repetitions,
                "sandbox_timeout": args.sandbox_timeout,
                "command_timeout": args.command_timeout,
                "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
            },
        )
    if args.provider == "tinker-chat":
        if args.renderer_name:
            raise ValueError(
                "Hosted Chat Completions uses the server renderer; omit --renderer-name"
            )
        completer = ChatAnnotationCompleter(
            model=args.checkpoint_url or args.model_name,
            max_tokens=args.max_tokens,
            thinking_effort=args.thinking_effort,
            temperature=args.temperature,
            base_url=args.base_url or TINKER_CHAT_BASE_URL,
            reports_dir=args.reports_dir / "api_calls",
        )
    elif args.provider == "openai":
        completer = OpenAIAnnotationCompleter(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.openai_model,
            max_output_tokens=args.max_tokens,
            reasoning_effort=args.openai_reasoning_effort,
        )
    else:
        service_client = tinker.ServiceClient(
            base_url=args.base_url,
            user_metadata=recipe_user_metadata("kokkos_auto_annotation"),
        )
        if args.checkpoint_url:
            sampling_client = service_client.create_sampling_client(
                model_path=args.checkpoint_url,
                base_model=args.model_name,
            )
        else:
            sampling_client = service_client.create_sampling_client(base_model=args.model_name)
        tokenizer = tokenizer_utils.get_tokenizer(args.model_name)
        renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(
            args.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)
        completer = TinkerAnnotationCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking_effort=args.thinking_effort,
        )

    instances = _read_instances(
        args.candidates,
        args.max_candidates,
        args.instance_id,
        offset=args.offset,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    async def run_one(instance: KokkosInstance) -> AutoAnnotationResult:
        report_path = args.reports_dir / f"{instance.instance_id}.json"
        if not args.no_resume and report_path.is_file():
            cached = AutoAnnotationResult.from_dict(json.loads(report_path.read_text()))
            if cached.passed and cached.validated_instance is not None:
                print(f"{instance.instance_id}: SKIP (already validated)")
                return cached
            if args.skip_failed_reports:
                print(f"{instance.instance_id}: SKIP (already rejected)")
                return cached
        async with semaphore:
            item = await annotate_and_validate_instance(
                instance,
                completer=completer,
                max_attempts=args.max_attempts,
                sandbox_timeout=args.sandbox_timeout,
                command_timeout=args.command_timeout,
                flaky_repetitions=args.flaky_repetitions,
                sandbox_factory=(
                    contree_validation_sandbox_factory
                    if args.sandbox_backend == "contree"
                    else default_validation_sandbox_factory
                ),
            )
            report_path.write_text(json.dumps(item.to_dict(), indent=2, sort_keys=True) + "\n")
            status = "PASS" if item.passed else "FAIL"
            print(f"{instance.instance_id}: {status} after {len(item.attempts)} attempt(s)")
            return item

    try:
        results = await asyncio.gather(*(run_one(instance) for instance in instances))
    finally:
        if isinstance(completer, ChatAnnotationCompleter):
            await completer.close()
    validated = [
        item.validated_instance.to_dict()
        for item in results
        if item.passed and item.validated_instance is not None
    ]
    _write_jsonl(args.output, validated)
    if args.failed_output:
        _write_jsonl(
            args.failed_output,
            [item.to_dict() for item in results if not item.passed],
        )
    print(f"validated {len(validated)}/{len(results)} candidates")


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
