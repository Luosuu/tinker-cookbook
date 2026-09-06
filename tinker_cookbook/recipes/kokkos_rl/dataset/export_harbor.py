"""Export validated Kokkos instances as hidden-test Harbor RL tasks."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.dataset.ecosystem import get_repository_profile
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.dataset.patching import (
    is_forbidden_agent_path,
    split_unified_diff,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.test_commands import (
    guard_source,
    normalize_test_command,
)

ERA_IMAGES = {
    "cpp14": "ubuntu:20.04",
    "cpp17": "ubuntu:22.04",
    "cpp20": "ubuntu:24.04",
}
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
HIP_IMAGE = "rocm/dev-ubuntu-22.04:6.4.3-complete"
SYCL_IMAGE = "intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04"
KOKKOS_DEPENDENCY_REF = "5.2.0"


def _clean_room_command(base_commit: str) -> str:
    return (
        "cd /workspace/repo"
        f" && git checkout -q --detach {shlex.quote(base_commit)}"
        f" && printf '%s\\n' {shlex.quote(base_commit)} > .git/shallow"
        " && git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d"
        " && git remote remove origin"
        " && git reflog expire --expire=now --all"
        " && git gc --prune=now -q"
    )


def _dockerfile(instance: KokkosInstance) -> str:
    profile = get_repository_profile(instance.repo)
    toolchain = str(instance.metadata.get("toolchain", instance.metadata.get("accelerator", "cpu")))
    image = {
        "cuda": CUDA_IMAGE,
        "hip": HIP_IMAGE,
        "sycl": SYCL_IMAGE,
    }.get(toolchain, ERA_IMAGES[instance.era])
    build_command = _build_command(instance)
    commands = [
        f"git clone https://github.com/{instance.repo}.git /workspace/repo",
        f"git -C /workspace/repo checkout {shlex.quote(instance.base_commit)}",
        "git -C /workspace/repo submodule update --init --recursive",
        # Retain the original base object as a shallow boundary. The verifier
        # pins this immutable SHA in its trusted script, independent of agent
        # commits, refs, or replacement objects. Prune all other history.
        _clean_room_command(instance.base_commit),
    ]
    if instance.repo != "kokkos/kokkos" and profile.build_system == "cmake":
        dependency_configure = (
            "cmake -S /workspace/kokkos -B /workspace/kokkos-build -G Ninja "
            "-DCMAKE_INSTALL_PREFIX=/opt/kokkos -DKokkos_ENABLE_SERIAL=ON "
            "-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF "
            "-DCMAKE_BUILD_TYPE=Release"
        )
        if toolchain == "cuda":
            dependency_configure += (
                " -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON "
                "-DCMAKE_CXX_COMPILER=/workspace/kokkos/bin/nvcc_wrapper"
            )
        elif toolchain == "hip":
            dependency_configure += (
                " -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc"
            )
        elif toolchain == "sycl":
            dependency_configure += (
                " -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_PVC=ON -DCMAKE_CXX_COMPILER=icpx"
            )
        commands.extend(
            [
                "git clone --branch "
                + shlex.quote(KOKKOS_DEPENDENCY_REF)
                + " --depth 1 https://github.com/kokkos/kokkos.git /workspace/kokkos",
                dependency_configure,
                "cmake --build /workspace/kokkos-build --parallel",
                "cmake --install /workspace/kokkos-build",
            ]
        )
    if profile.build_system == "python":
        commands.extend(
            [
                "python -m pip install --break-system-packages "
                "numpy patchelf pybind11 pytest setuptools wheel",
                "cd /workspace/repo && PIP_BREAK_SYSTEM_PACKAGES=1 "
                "python install_base.py install -- "
                "-DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF "
                "-DENABLE_VIEW_RANKS=4 -DENABLE_CUDA=OFF "
                "-DENABLE_THREADS=OFF -DENABLE_OPENMP=ON",
            ]
        )
    commands.extend(
        [
            f"cd /workspace/repo && {instance.configure_command}",
            f"cd /workspace/repo && {build_command}",
        ]
    )
    setup = (" " + "\\" + "\n    && ").join(commands)
    return f"""FROM {image}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \\
    ninja-build openmpi-bin python-is-python3 python3-pip \\
    && rm -rf /var/lib/apt/lists/*
RUN {setup}
WORKDIR /workspace/repo
"""


def _build_command(instance: KokkosInstance) -> str:
    if instance.build_command:
        return instance.build_command
    targets = " ".join(shlex.quote(target) for target in instance.build_targets)
    return f"cmake --build build --target {targets} --parallel"


def _shell_array(name: str, commands: tuple[str, ...]) -> str:
    values = " ".join(shlex.quote(command) for command in commands)
    return f"{name}=({values})"


def _test_script(instance: KokkosInstance) -> str:
    build_command = _build_command(instance)
    # Bare --parallel overrides CMAKE_BUILD_PARALLEL_LEVEL with the native
    # default. Pass the intended job count explicitly to avoid verifier OOMs.
    # Keep custom commands with explicit parallelism unchanged.
    if build_command.endswith(" --parallel"):
        build_command += ' "${CMAKE_BUILD_PARALLEL_LEVEL:-1}"'
    protected_paths = tuple(path for path, _block in split_unified_diff(instance.test_patch))
    protected_values = " ".join(shlex.quote(path) for path in protected_paths)
    return f"""#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo" || exit 0
baseline={shlex.quote(instance.base_commit)}

# Never trust HEAD: the agent may have committed its edits.
# Fail closed if the original object was deleted; ignore replacement refs.
git --no-replace-objects cat-file -e "$baseline^{{commit}}" || exit 0
illegal=$(\n  set -e\n  git --no-replace-objects diff --no-ext-diff --name-only "$baseline" -- || exit 1\n  git ls-files --others --exclude-standard | sed '\\#^build/#d'\n) || exit 0
if printf '%s\n' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\\.py)$|(^|/)CMakeLists\\.txt$|^\\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=({protected_values})
for path in "${{protected_paths[@]}}"; do
  if git --no-replace-objects cat-file -e "$baseline:$path" 2>/dev/null; then
    git --no-replace-objects checkout "$baseline" -- "$path" || exit 0
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! {build_command}; then
  exit 0
fi

run_checked() {{
  python3 - "$1" <<'KOKKOS_RUNTIME_GUARD'
{guard_source()}
KOKKOS_RUNTIME_GUARD
}}

{_shell_array("f2p_commands", tuple(normalize_test_command(command, test_patch=instance.test_patch) for command in instance.f2p_commands))}
{_shell_array("p2p_commands", tuple(normalize_test_command(command, test_patch=instance.test_patch) for command in instance.p2p_commands))}
for command in "${{f2p_commands[@]}}" "${{p2p_commands[@]}}"; do
  if [[ -n "$command" ]] && ! run_checked "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
"""


def _solution_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
git apply --whitespace=nowarn /solution/gold.patch
"""


def _agent_allowed_code_patch(instance: KokkosInstance) -> str:
    """Return only production changes that the verifier permits an agent to make."""

    blocks = split_unified_diff(instance.code_patch)
    if not blocks:
        return instance.code_patch
    return "".join(
        block for path, block in blocks if not is_forbidden_agent_path(path, instance.repo)
    )


def _task_toml(instance: KokkosInstance, org: str) -> str:
    requires_gpu = bool(instance.metadata.get("requires_gpu", False))
    gpu_lines = 'gpus = 1\ngpu_types = ["L4"]\n' if requires_gpu else ""
    description = f"Repair {instance.repo} PR-derived regression {instance.pr_number}"
    return f"""schema_version = "1.4"
artifacts = []

[task]
name = {json.dumps(f"{org}/{instance.instance_id}")}
version = "1.0.0"
description = {json.dumps(description)}
authors = [{{ name = "SWE-kokkos-bench contributors" }}]
keywords = ["kokkos", "swe-bench", "coding-agent"]

[metadata]
difficulty = {json.dumps(str(instance.metadata.get("difficulty", "unknown")))}
repository = {json.dumps(instance.repo)}
pr_number = {instance.pr_number}

[verifier]
timeout_sec = 1200.0
network_mode = "no-network"

[agent]
timeout_sec = 3600.0
network_mode = "allowlist"
allowed_hosts = ["api.openai.com"]

[environment]
network_mode = "public"
build_timeout_sec = 3600.0
os = "linux"
cpus = 4
memory_mb = 16384
{gpu_lines}workdir = "/workspace/repo"
"""


def _instruction(instance: KokkosInstance) -> str:
    return f"""Fix the following issue in the Kokkos repository.

{instance.problem_statement.strip()}

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Network access is unavailable and git history contains
only the base revision, so do not spend time looking for upstream commits or
pull requests. Diagnose from the local source, make a production-code change,
and use the smallest relevant target in the existing build tree for local
verification.
"""


def export_instance(instance: KokkosInstance, output_dir: Path, *, org: str = "swe-kokkos") -> Path:
    if not instance.is_validation_ready:
        raise ValueError(f"{instance.instance_id} has not been annotated for validation")
    task_dir = output_dir / instance.instance_id
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "solution").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text(_dockerfile(instance))
    (task_dir / "tests" / "test.sh").write_text(_test_script(instance))
    (task_dir / "tests" / "test.patch").write_text(instance.test_patch)
    allowed_code_patch = _agent_allowed_code_patch(instance)
    if not allowed_code_patch.strip():
        raise ValueError(f"{instance.instance_id} has no agent-allowed production patch")
    (task_dir / "solution" / "gold.patch").write_text(allowed_code_patch)
    (task_dir / "solution" / "solve.sh").write_text(_solution_script())
    (task_dir / "instruction.md").write_text(_instruction(instance))
    (task_dir / "metadata.json").write_text(
        json.dumps(instance.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (task_dir / "task.toml").write_text(_task_toml(instance, org))
    return task_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--org", default="swe-kokkos")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = 0
    for line in args.instances.read_text().splitlines():
        if not line.strip():
            continue
        export_instance(KokkosInstance.from_dict(json.loads(line)), args.output_dir, org=args.org)
        count += 1
    print(f"exported {count} Harbor tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
