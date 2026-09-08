"""Audited donor-only SFT arms; no sampling, sandbox work, or implicit recovery."""

from __future__ import annotations

import asyncio
import fcntl
import importlib.metadata
import json
import math
import os
import traceback
from pathlib import Path
from typing import cast

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.recipes.kokkos_rl.rl.donor_qualification import (
    names,
    prepare,
    proof_path,
    records,
)
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_openai import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.nebius_phase_ledger import pinned_file
from tinker_cookbook.recipes.kokkos_rl.rl.rollout_data import RecordedRollout
from tinker_cookbook.recipes.kokkos_rl.rl.saved_candidate_regrade import (
    ProofArchive,
    digest,
    exclusive_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.self_train import (
    RecordedDataset,
    choose_donors,
    write_json,
)
from tinker_cookbook.recipes.kokkos_rl.rl.validation_recovery import mapping, read_json
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

ARMS = ("random_success", "short_success")
EXCLUDED = "kokkos__kokkos-7148__02"
PROTOCOL = {
    "model_name": "thinkingmachines/Inkling-Small:peft:262144",
    "lora_rank": 32,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "lr_schedule": "constant",
    "tasks_per_batch": 4,
    "max_updates_per_arm": 20,
    "actual_updates_per_arm": 8,
    "total_updates": 16,
    "max_turns": 40,
    "max_response_tokens": 16384,
    "max_sampled_tokens": 65536,
    "max_datum_tokens": 114688,
    "thinking_effort": 0.9,
    "selection_seed": 7,
    "epoch_shuffle_seed": 0,
    "submit_ahead": 1,
    "optimizer": "Adam",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "training_weighting": "sum of supervised weights is one per trajectory",
    "inference_requests": 0,
    "sandbox_requests": 0,
    "automatic_evaluation": False,
}


def code_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[4]
    files = [
        "recipes/kokkos_rl/rl/donor_sft.py",
        "recipes/kokkos_rl/rl/self_train.py",
        "recipes/kokkos_rl/rl/rollout_data.py",
        "recipes/kokkos_rl/rl/donor_qualification.py",
        "recipes/kokkos_rl/rl/saved_candidate_regrade.py",
        "recipes/kokkos_rl/rl/validation_recovery.py",
        "recipes/kokkos_rl/rl/nebius_phase_ledger.py",
        "supervised/train.py",
        "supervised/types.py",
        "supervised/common.py",
        "supervised/data.py",
        "checkpoint_utils.py",
        "model_info.py",
        "tokenizer_utils.py",
        "utils/ml_log.py",
        "utils/lr_scheduling.py",
        "stores/training_store.py",
    ]
    return {name: pinned_file(repo / "tinker_cookbook" / name)["sha256"] for name in files}


class PinnedDataset(RecordedDataset):
    def __init__(self, paths: list[str], proofs: dict[str, object]):
        super().__init__(paths, 4)
        self.proofs = proofs

    def get_batch(self, index: int) -> list[tinker.Datum]:
        if not 0 <= index < len(self):
            raise ValueError("Batch index exceeds fixed one-epoch schedule")
        for pos in self.order[index * self.batch_size : (index + 1) * self.batch_size]:
            proof_path(mapping(self.proofs[self.paths[pos]]))
        return super().get_batch(index)


@chz.chz
class PinnedDatasetBuilder(SupervisedDatasetBuilder):
    manifest: str
    manifest_sha256: str

    def __call__(self) -> tuple[SupervisedDataset, None]:
        p = proof_path({"path": str(Path(self.manifest).resolve()), "sha256": self.manifest_sha256})
        content = read_json(p)
        paths = content["paths"]
        if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
            raise ValueError("Invalid explicit donor paths")
        train_names, heldout = names(content["train_tasks"]), names(content["heldout_tasks"])
        seen = set()
        for path in paths:
            proof_path(mapping(mapping(content["proofs"])[path]))
            row = read_json(Path(path))
            task = str(row["task_name"])
            if task not in train_names or task in heldout or task in seen:
                raise ValueError("Heldout or repeated donor task")
            seen.add(task)
            for datum in records(row["datums"]):
                for key in ("input_tokens", "target_tokens"):
                    xs = datum[key]
                    if not isinstance(xs, list) or any(type(x) is not int or x < 0 for x in xs):
                        raise ValueError("Noninteger raw training token")
        if len(paths) != 31 or len(seen) != 31:
            raise ValueError("Exactly31 distinct reviewed tasks are required")
        return PinnedDataset(paths, mapping(content["proofs"])), None


def selection(
    rows: list[tuple[Path, RecordedRollout]], review: dict[str, object]
) -> dict[str, list[str]]:
    if (
        review.get("status") != "accepted_donor_selection_only"
        or review.get("accepted_candidates") != 90
    ):
        raise ValueError("Explicit90-donor selection review required")
    if set(mapping(review["excluded_slots"])) != {EXCLUDED}:
        raise ValueError("Excluded donor set changed")
    slots = [f"{row['task_name']}__{row['sample_index']:02d}" for _, row in rows]
    if len(slots) != 91 or len(set(slots)) != 91 or slots.count(EXCLUDED) != 1:
        raise ValueError("Exact91 unique source candidates including the excluded slot required")
    accepted = [(p, r) for p, r in rows if f"{r['task_name']}__{r['sample_index']:02d}" != EXCLUDED]
    if len(accepted) != 90:
        raise ValueError("Expected90 accepted candidates")
    arms = choose_donors(accepted)
    for arm, paths in arms.items():
        selected = [read_json(Path(p)) for p in paths]
        actual = {
            "slots": [f"{r['task_name']}__{int(str(r['sample_index'])):02d}" for r in selected],
            "tasks": len(selected),
            "turns": sum(int(str(r["turns"])) for r in selected),
            "sampled_tokens": sum(int(str(r["sampled_tokens"])) for r in selected),
            "datums": sum(len(records(r["datums"])) for r in selected),
            "updates": math.ceil(len(paths) / 4),
        }
        if actual != mapping(mapping(review["arms"])[arm]):
            raise ValueError("Selection differs from exact reviewed slots/budget")
        if actual["tasks"] != 31 or actual["updates"] != 8:
            raise ValueError("Task support or update budget changed")
    return arms


def check_collection_protocol(config: dict[str, object], evaluation: dict[str, object]) -> None:
    expected = {
        "model_name": PROTOCOL["model_name"],
        "max_turns": 40,
        "train_samples": 4,
        "eval_samples": 4,
        "eval_size": 20,
        "split_seed": 7,
        "learning_rate": 1e-5,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Original collection protocol differs from SFT continuation")
    expected_eval = {
        "model_name": PROTOCOL["model_name"],
        "max_turns": 40,
        "max_tokens": 16384,
        "max_sampled_tokens": 65536,
        "max_trajectory_tokens": 114688,
        "thinking_effort": 0.9,
        "temperature": 1.0,
    }
    if any(evaluation.get(key) != value for key, value in expected_eval.items()):
        raise ValueError("Original inference identity differs")


def create_bundle(
    collection: Path, qualification: Path, review_path: Path, bundle: Path
) -> dict[str, object]:
    if (bundle / "manifest.json").exists():
        return verify_bundle(bundle)
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError("Incomplete bundle preparation retained; use a new explicit directory")
    review = read_json(review_path)
    raw = collection / "donor_raw_audit_v2/report.json"
    final = qualification / "monitor/final_snapshot_v2/report.json"
    if (
        pinned_file(raw)["sha256"] != review["raw_audit_sha256"]
        or pinned_file(final)["sha256"] != review["qualification_audit_sha256"]
    ):
        raise ValueError("Reviewed audit changed")
    inputs, _ = prepare(collection, raw)
    eval_path = collection / "collection/eval_identity.json"
    check_collection_protocol(
        read_json(collection / "config.json"),
        mapping(mapping(read_json(eval_path)["identity"])["config"]),
    )
    state = read_json(collection / "status.json")
    if state.get("stage") != "blocked":
        raise ValueError("Original blocked collection status changed")
    result_rows = [
        json.loads(x) for x in (collection / "collection/results.jsonl").read_text().splitlines()
    ]
    errors = [
        {"task": r["task_name"], "sample": r["sample_index"], "error": r["error"]}
        for r in result_rows
        if r["error"] is not None
    ]
    if len(result_rows) != 320 or len(errors) != 13:
        raise ValueError("Original error inventory changed")
    bad = collection / "collection/rollouts/kokkos__kokkos-7043__03/candidate.json"
    if read_json(bad).get("complete") is not False:
        raise ValueError("Known incomplete capture evidence changed")
    if read_json(qualification / "status.json").get("stage") != "awaiting_donor_review":
        raise ValueError("Fresh qualification incomplete")
    qidentity = read_json(qualification / "identity.json")
    if mapping(qidentity["inputs"]) != inputs:
        raise ValueError("Qualification input differs from audited collection")
    qsha = digest(qidentity)
    candidates = []
    for row in records(inputs["records"]):
        folder = qualification / str(row["slot"])
        claim = read_json(folder / "claim.json")
        result = read_json(folder / "result.json")
        if (
            claim.get("identity_sha256") != qsha
            or claim.get("slot") != row["slot"]
            or claim.get("patch_sha256") != mapping(row["patch_proof"])["sha256"]
        ):
            raise ValueError("Qualification claim differs")
        if (
            result.get("claim") != claim
            or result.get("reward") != 1
            or result.get("error") is not None
            or result.get("stage") != "complete"
        ):
            raise ValueError("Fresh qualification did not pass")
        p = proof_path(mapping(row["trajectory_proof"]))
        candidates.append((p, cast(RecordedRollout, read_json(p))))
    arms = selection(candidates, review)
    archive = ProofArchive(bundle)
    # Freeze the completed grader audit and all evidence it asserts, including source.
    report = read_json(final)
    for proof in records(report["proofs"]):
        original = Path(str(proof["source"]))
        obj = final.parent / str(proof["object"])
        if (
            pinned_file(original)["sha256"] != proof["sha256"]
            or pinned_file(obj)["sha256"] != proof["sha256"]
        ):
            raise ValueError("Completed qualification evidence changed")
        archive.add({"path": str(original), "sha256": proof["sha256"]})
    for path in (raw, final, review_path, collection / "status.json", bad, eval_path):
        archive.add(mapping(pinned_file(path)))
    arm_proofs = {}
    for arm, paths in arms.items():
        frozen = [archive.add(mapping(pinned_file(Path(p)))) for p in paths]
        manifest = {
            "paths": [r["path"] for r in frozen],
            "proofs": {r["path"]: r for r in frozen},
            "train_tasks": inputs["train"],
            "heldout_tasks": inputs["heldout"],
            "slots": mapping(mapping(review["arms"])[arm])["slots"],
            "protocol": PROTOCOL,
        }
        exclusive_json(bundle / f"{arm}.json", manifest)
        arm_proofs[arm] = pinned_file(bundle / f"{arm}.json")
    manifest = {
        "version": 1,
        "purpose": "donor_only_sft_not_collection_recovery",
        "protocol": PROTOCOL,
        "selection_review": archive.add(mapping(pinned_file(review_path))),
        "arms": arm_proofs,
        "train": inputs["train"],
        "heldout": inputs["heldout"],
        "qualification": inputs["qualification"],
        "original_collection": {
            "eval_identity": pinned_file(eval_path),
            "manifest": inputs["collection_manifest"],
            "results": inputs["original_results"],
            "status": pinned_file(collection / "status.json"),
            "errors": errors,
            "capture_exclusion": pinned_file(bad),
            "unchanged_blocked": True,
        },
        "archive": archive.entries,
        "code": code_identity(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tinker", "numpy", "torch", "tml-renderers")
        },
    }
    exclusive_json(bundle / "manifest.json", manifest)
    return verify_bundle(bundle)


def verify_bundle(bundle: Path) -> dict[str, object]:
    m = read_json(bundle / "manifest.json")
    if m["protocol"] != PROTOCOL or m["code"] != code_identity():
        raise ValueError("Protocol or training source changed")
    if m["versions"] != {
        name: importlib.metadata.version(name)
        for name in ("tinker", "numpy", "torch", "tml-renderers")
    }:
        raise ValueError("Training dependencies changed")
    archive = ProofArchive(bundle, mapping(m["archive"]))
    archive.verify()
    original = mapping(m["original_collection"])
    for key in ("manifest", "results", "status", "capture_exclusion", "eval_identity"):
        proof_path(mapping(original[key]))
    if read_json(proof_path(mapping(original["status"]))).get("stage") != "blocked":
        raise ValueError("Original state must remain blocked")
    for arm in ARMS:
        proof = mapping(mapping(m["arms"])[arm])
        path = proof_path(proof)
        dataset, _ = PinnedDatasetBuilder(
            manifest=str(path), manifest_sha256=str(proof["sha256"])
        )()
        if len(dataset) != 8:
            raise ValueError("Training step count changed")
        for i in range(len(dataset)):
            dataset.get_batch(i)
    return m


def training_config(bundle: Path, arm: str, log: Path) -> train.Config:
    m = read_json(bundle / "manifest.json")
    proof = mapping(mapping(m["arms"])[arm])
    return train.Config(
        log_path=str(log),
        model_name=str(PROTOCOL["model_name"]),
        recipe_name="kokkos_donor_only_sft",
        renderer_name=model_info.get_recommended_renderer_name(str(PROTOCOL["model_name"])),
        dataset_builder=PinnedDatasetBuilder(
            manifest=str(proof["path"]), manifest_sha256=str(proof["sha256"])
        ),
        learning_rate=1e-5,
        lora_rank=32,
        num_epochs=1,
        lr_schedule="constant",
        max_steps=8,
        save_every=0,
        save_every_tokens=0,
        save_every_seconds=0,
        rolling_save_every=0,
        eval_every=0,
        infrequent_eval_every=0,
        wandb_project=None,
        submit_ahead=1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )


def training_evidence(log: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    cp = log / "checkpoints.jsonl"
    checkpoints = [json.loads(x) for x in cp.read_text().splitlines()]
    if (
        len(checkpoints) != 1
        or checkpoints[0].get("name") != "final"
        or checkpoints[0].get("epoch") != 1
        or checkpoints[0].get("batch") != 0
    ):
        raise ValueError("Missing exact one-epoch final checkpoint evidence")
    final = checkpoints[0]
    if not all(
        str(final.get(k, "")).startswith("tinker://") for k in ("sampler_path", "state_path")
    ):
        raise ValueError("Final state/sampler checkpoint missing")
    metrics_path = log / "metrics.jsonl"
    metrics = [json.loads(x) for x in metrics_path.read_text().splitlines()]
    if [r.get("step") for r in metrics] != list(range(8)) or any(
        r.get("epoch") != 0 or r.get("learning_rate") != 1e-5 for r in metrics
    ):
        raise ValueError("Training did not record exactly8 fixed-budget updates")
    if abs(sum(r["num_loss_tokens"] for r in metrics) - 31) > 1e-4:
        raise ValueError("Training trajectory weighting differs")
    return final, [pinned_file(cp), pinned_file(metrics_path)]


def prior_arm(folder: Path, claim: dict[str, object]) -> dict[str, object] | None:
    marker = folder / "claim.json"
    result = folder / "result.json"
    if marker.exists():
        if read_json(marker) != claim:
            raise ValueError("Arm claim identity changed")
        if not result.exists():
            raise ValueError(
                "Interrupted training needs explicit recovery review; no ordinary resume"
            )
        row = read_json(result)
        if (
            row.get("claim") != claim
            or row.get("stage") != "complete"
            or row.get("error") is not None
        ):
            raise ValueError("Prior training failure requires recovery review")
        checkpoint, proofs = training_evidence(folder / "training")
        if row.get("checkpoint") != checkpoint or row.get("proofs") != proofs:
            raise ValueError("Completed training receipt differs from actual logs")
        for p in records(row["proofs"]):
            proof_path(p)
        return row
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Unclaimed training artifacts cannot be resumed")
    return None


async def train_arm(
    bundle: Path,
    arm: str,
    folder: Path,
    identity_sha: str,
    bundle_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    claim = {
        "identity_sha256": identity_sha,
        "arm": arm,
        "updates": 8,
        "epochs": 1,
        "fresh_base": True,
        "automatic_resume": False,
    }
    if bundle_manifest is not None:
        proof_path(bundle_manifest)
    prior = prior_arm(folder, claim)
    if prior is not None:
        return prior
    verify_bundle(bundle)
    exclusive_json(folder / "claim.json", claim)
    result: dict[str, object] = {"claim": claim}
    try:
        # This directory must be absent: train.main's implicit checkpoint resume
        # is never entered by this once-only wrapper.
        log = folder / "training"
        if log.exists():
            raise ValueError("Training log already exists")
        await train.main(training_config(bundle, arm, log))
        if bundle_manifest is not None:
            proof_path(bundle_manifest)
        verify_bundle(bundle)
        final, proofs = training_evidence(log)
        result.update(stage="complete", error=None, checkpoint=final, proofs=proofs)

    except BaseException as error:
        result.update(
            stage="blocked",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        exclusive_json(folder / "result.json", result)
        raise
    exclusive_json(folder / "result.json", result)
    return result


@chz.chz
class Config:
    collection_dir: str
    qualification_dir: str
    selection_review: str
    bundle_dir: str
    output_dir: str
    dispatch: bool = False
    training_review_receipt: str | None = None
    env_file: str = ".env"


async def run(config: Config) -> None:
    collection, qualification, bundle, output = (
        Path(x).resolve()
        for x in (
            config.collection_dir,
            config.qualification_dir,
            config.bundle_dir,
            config.output_dir,
        )
    )
    if (
        any(output == x or output.is_relative_to(x) for x in (collection, qualification, bundle))
        or bundle == output
    ):
        raise ValueError("Execution output must be separate from immutable inputs")
    m = create_bundle(collection, qualification, Path(config.selection_review).resolve(), bundle)
    identity = {
        "bundle": pinned_file(bundle / "manifest.json"),
        "protocol": PROTOCOL,
        "code": m["code"],
        "versions": m["versions"],
        "arms": list(ARMS),
        "output_dir": str(output),
        "automatic_eval": False,
    }
    identity_sha = digest(identity)
    for arm in ARMS:
        prior_arm(
            output / arm,
            {
                "identity_sha256": identity_sha,
                "arm": arm,
                "updates": 8,
                "epochs": 1,
                "fresh_base": True,
                "automatic_resume": False,
            },
        )
    if not config.dispatch:
        print(
            json.dumps(
                {
                    "identity_sha256": identity_sha,
                    "identity": identity,
                    "updates_per_arm": 8,
                    "total_updates": 16,
                    "model_requests": 0,
                    "sandbox_requests": 0,
                }
            )
        )
        return
    if config.training_review_receipt is None or read_json(
        Path(config.training_review_receipt)
    ) != {
        "status": "accepted_training_only",
        "identity_sha256": identity_sha,
        "max_updates_total": 16,
    }:
        raise ValueError("Exact training-only review required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "identity.json").exists():
            if read_json(output / "identity.json") != identity:
                raise ValueError("Training phase identity changed")
        else:
            exclusive_json(output / "identity.json", identity)
        load_env_file(Path(config.env_file))
        results = []
        try:
            for arm in ARMS:
                write_json(
                    output / "status.json",
                    {
                        "stage": "training",
                        "arm": arm,
                        "identity_sha256": identity_sha,
                        "pid": os.getpid(),
                    },
                )
                results.append(
                    await train_arm(
                        bundle, arm, output / arm, identity_sha, mapping(identity["bundle"])
                    )
                )
        except BaseException as error:
            write_json(
                output / "status.json",
                {
                    "stage": "blocked",
                    "error": f"{type(error).__name__}: {error}",
                    "identity_sha256": identity_sha,
                },
            )
            raise
        write_json(output / "results.json", results)
        write_json(
            output / "status.json",
            {
                "stage": "awaiting_checkpoint_review",
                "arms_completed": 2,
                "identity_sha256": identity_sha,
                "automatic_eval": False,
            },
        )


if __name__ == "__main__":
    asyncio.run(chz.entrypoint(run))
