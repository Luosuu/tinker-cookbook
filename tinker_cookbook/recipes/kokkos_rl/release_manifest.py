"""Build a digest-bound Oracle/NOP validation manifest for a Harbor dataset."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialEvidence:
    agent: str
    digest: str
    path: Path
    reward: float


def _dataset_digests(dataset_dir: Path) -> dict[str, str]:
    manifest = tomllib.loads((dataset_dir / "dataset.toml").read_text())
    return {
        str(item["name"]).split("/", maxsplit=1)[-1]: str(item["digest"])
        for item in manifest["tasks"]
    }


def _instance_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["instance_id"])
        for line in path.read_text().splitlines()
        if line.strip()
    }


def _collect_evidence(roots: list[Path]) -> dict[tuple[str, str, str], TrialEvidence]:
    evidence: dict[tuple[str, str, str], TrialEvidence] = {}
    for root in roots:
        for result_path in root.glob("**/result.json"):
            value = json.loads(result_path.read_text())
            task_name = value.get("task_name")
            agent_info = value.get("agent_info")
            verifier_result = value.get("verifier_result")
            if not isinstance(task_name, str) or not isinstance(agent_info, dict):
                continue
            if not isinstance(verifier_result, dict):
                continue
            rewards = verifier_result.get("rewards")
            if not isinstance(rewards, dict) or not isinstance(rewards.get("reward"), int | float):
                continue
            lock = json.loads((result_path.parent / "lock.json").read_text())
            task = lock.get("task")
            if not isinstance(task, dict):
                continue
            agent = str(agent_info.get("name", ""))
            digest = str(task.get("digest", ""))
            instance_id = task_name.split("/", maxsplit=1)[-1]
            item = TrialEvidence(agent, digest, result_path, float(rewards["reward"]))
            evidence[(instance_id, agent, digest)] = item
    return evidence


def build_manifest(
    dataset_dir: Path,
    old_instances: Path,
    evidence_roots: list[Path],
) -> dict[str, object]:
    digests = _dataset_digests(dataset_dir)
    old_ids = _instance_ids(old_instances)
    construction = json.loads((dataset_dir / "manifest.json").read_text())
    construction_by_id = {
        str(item["instance_id"]): str(item["validation_source"])
        for item in construction["instances"]
    }
    evidence = _collect_evidence(evidence_roots)
    instances: list[dict[str, object]] = []
    for instance_id, registry_digest in sorted(digests.items()):
        validation_digests = {
            digest
            for candidate_id, agent, digest in evidence
            if candidate_id == instance_id
            and agent == "oracle"
            and evidence[(candidate_id, agent, digest)].reward == 1.0
            and (nop := evidence.get((instance_id, "nop", digest))) is not None
            and nop.reward == 0.0
        }
        if len(validation_digests) != 1:
            raise ValueError(
                f"expected one Oracle/NOP validation digest for {instance_id}, "
                f"found {sorted(validation_digests)}"
            )
        validation_digest = validation_digests.pop()
        oracle = evidence[(instance_id, "oracle", validation_digest)]
        nop = evidence[(instance_id, "nop", validation_digest)]
        instances.append(
            {
                "instance_id": instance_id,
                "registry_digest": registry_digest,
                "validation_digest": validation_digest,
                "release_group": "v1-unchanged" if instance_id in old_ids else "v2-addition",
                "construction_source": construction_by_id[instance_id],
                "oracle": {"reward": oracle.reward, "result": str(oracle.path)},
                "nop": {"reward": nop.reward, "result": str(nop.path)},
            }
        )
    return {
        "dataset": "luosuu/SWE-kokkos-bench",
        "version": "2.0.0",
        "instance_count": len(instances),
        "oracle_passed": sum(item["oracle"]["reward"] == 1.0 for item in instances),
        "nop_rejected": sum(item["nop"]["reward"] == 0.0 for item in instances),
        "instances": instances,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--old-instances", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manifest(args.dataset_dir, args.old_instances, args.evidence_root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {manifest['instance_count']} digest-bound validation records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
