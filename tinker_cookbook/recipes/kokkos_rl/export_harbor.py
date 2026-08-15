"""Export validated Kokkos instances as hidden-test Harbor RL tasks."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.models import KokkosInstance

ERA_IMAGES = {
    "cpp14": "ubuntu:20.04",
    "cpp17": "ubuntu:22.04",
    "cpp20": "ubuntu:24.04",
}


def _dockerfile(instance: KokkosInstance) -> str:
    image = ERA_IMAGES[instance.era]
    targets = " ".join(shlex.quote(target) for target in instance.build_targets)
    return f"""FROM {image}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential ca-certificates ccache cmake git ninja-build \\
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/{instance.repo}.git /workspace/repo \\
    && cd /workspace/repo \\
    && git checkout {shlex.quote(instance.base_commit)}
WORKDIR /workspace/repo
RUN {instance.configure_command}
RUN cmake --build build --target {targets} --parallel
"""


def _shell_array(name: str, commands: tuple[str, ...]) -> str:
    values = " ".join(shlex.quote(command) for command in commands)
    return f"{name}=({values})"


def _test_script(instance: KokkosInstance) -> str:
    targets = " ".join(shlex.quote(target) for target in instance.build_targets)
    return f"""#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(git diff --name-only {shlex.quote(instance.base_commit)} --)
if printf '%s\n' "$illegal" | grep -E '(^|/)(unit_tests?|CMakeLists\\.txt)(/|$)|^\\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

for path in core/unit_test containers/unit_tests algorithms/unit_tests; do
  if git cat-file -e {shlex.quote(instance.base_commit)}:"$path" 2>/dev/null; then
    git checkout {shlex.quote(instance.base_commit)} -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target {targets} --parallel; then
  exit 0
fi

{_shell_array('f2p_commands', instance.f2p_commands)}
{_shell_array('p2p_commands', instance.p2p_commands)}
for command in "${{f2p_commands[@]}}" "${{p2p_commands[@]}}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
"""


def _instruction(instance: KokkosInstance) -> str:
    return f"""Fix the following issue in the Kokkos repository.

{instance.problem_statement.strip()}

The repository is checked out at `/workspace/repo`. Work only on production
source code. Do not modify tests, CMake registration, CI configuration, or the
grading environment. Use the existing build tree for local verification.
"""


def export_instance(instance: KokkosInstance, output_dir: Path) -> Path:
    if not instance.is_validation_ready:
        raise ValueError(f"{instance.instance_id} has not been annotated for validation")
    task_dir = output_dir / instance.instance_id
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "solution").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text(_dockerfile(instance))
    (task_dir / "tests" / "test.sh").write_text(_test_script(instance))
    (task_dir / "tests" / "test.patch").write_text(instance.test_patch)
    (task_dir / "solution" / "gold.patch").write_text(instance.patch)
    (task_dir / "instruction.md").write_text(_instruction(instance))
    (task_dir / "metadata.json").write_text(
        json.dumps(instance.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (task_dir / "task.toml").write_text(
        f'version = "1.0"\nname = {json.dumps(instance.instance_id)}\n'
        f'difficulty = {json.dumps(str(instance.metadata.get("difficulty", "unknown")))}\n'
    )
    return task_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = 0
    for line in args.instances.read_text().splitlines():
        if not line.strip():
            continue
        export_instance(KokkosInstance.from_dict(json.loads(line)), args.output_dir)
        count += 1
    print(f"exported {count} Harbor tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
