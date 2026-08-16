"""Assemble deduplicated, validated Kokkos instances into a SWE-bench JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinker_cookbook.recipes.kokkos_rl.models import KokkosInstance


def assemble_instances(paths: list[Path]) -> tuple[list[KokkosInstance], dict[str, str]]:
    """Read validated sources in priority order and deduplicate by instance ID."""

    instances: dict[str, KokkosInstance] = {}
    provenance: dict[str, str] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            instance = KokkosInstance.from_dict(json.loads(line))
            if not instance.is_validation_ready:
                raise ValueError(
                    f"{path}:{line_number}: {instance.instance_id} is not validation-ready"
                )
            if instance.instance_id not in instances:
                instances[instance.instance_id] = instance
                provenance[instance.instance_id] = str(path)
    ordered = sorted(instances.values(), key=lambda item: (item.repo, item.pr_number))
    return ordered, provenance


def write_dataset(
    instances: list[KokkosInstance],
    provenance: dict[str, str],
    output: Path,
    manifest: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in instances)
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "format": "SWE-bench-compatible JSONL",
                "instance_count": len(instances),
                "instances": [
                    {
                        "instance_id": item.instance_id,
                        "repo": item.repo,
                        "validation_source": provenance[item.instance_id],
                    }
                    for item in instances
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    instances, provenance = assemble_instances(args.instances)
    write_dataset(instances, provenance, args.output, args.manifest)
    print(f"assembled {len(instances)} validated instances into {args.output}")


if __name__ == "__main__":
    main()
