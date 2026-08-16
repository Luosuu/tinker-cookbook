"""Merge reviewed verification annotations into mined Kokkos candidate JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tinker_cookbook.recipes.kokkos_rl.models import KokkosInstance

TUPLE_FIELDS = {
    "build_targets",
    "fail_to_pass",
    "pass_to_pass",
    "f2p_commands",
    "p2p_commands",
}

ANNOTATION_FIELDS = TUPLE_FIELDS | {"build_command", "configure_command", "metadata"}


def apply_annotations(instance: KokkosInstance, annotations: dict[str, Any]) -> KokkosInstance:
    unknown = set(annotations) - ANNOTATION_FIELDS
    if unknown:
        raise ValueError(f"unsupported annotation fields: {sorted(unknown)}")
    updates = dict(annotations)
    for field in TUPLE_FIELDS.intersection(updates):
        if not isinstance(updates[field], list):
            raise ValueError(f"annotation field {field!r} must be a JSON array")
        updates[field] = tuple(str(item) for item in updates[field])
    if "metadata" in updates:
        if not isinstance(updates["metadata"], dict):
            raise ValueError("annotation field 'metadata' must be a JSON object")
        updates["metadata"] = {**instance.metadata, **dict(updates["metadata"])}
    return replace(instance, **updates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    annotations = json.loads(args.annotations.read_text())
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for line in args.candidates.read_text().splitlines():
            if not line.strip():
                continue
            instance = KokkosInstance.from_dict(json.loads(line))
            values = annotations.get(instance.instance_id)
            if values is None:
                continue
            annotated = apply_annotations(instance, values)
            handle.write(json.dumps(annotated.to_dict(), sort_keys=True) + "\n")
            written += 1
    print(f"wrote {written} annotated instances")


if __name__ == "__main__":
    main()
