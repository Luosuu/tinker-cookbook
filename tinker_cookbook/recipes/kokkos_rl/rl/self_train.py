"""Gate a fixed-budget Kokkos RFT comparison on verifier and baseline integrity.

Reuses Harbor evaluation and the standard supervised trainer. All remote stages
are bounded, resumable, and recorded under one experiment directory.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import math
import random
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import cast

import chz
import numpy as np
import tinker

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, TaskResult, run_eval
from tinker_cookbook.recipes.harbor_rl.eval_state import _task_digest
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborTask,
    SandboxFactory,
    default_sandbox_factory,
    load_harbor_tasks_from_dir,
)
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
from tinker_cookbook.recipes.kokkos_rl.rl.rollout_data import RecordedRollout, eligible
from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder


@chz.chz
class Config:
    output_dir: str
    tasks_dir: str = "data/kokkos/SWE-kokkos-bench-v2"
    model_name: str = "thinkingmachines/Inkling-Small:peft:262144"
    sandbox_backend: str = "contree"
    cache_path: str = "notes/experiments/SWE-kokkos-bench/v2-pass-at-k/contree_images.json"
    concurrency: int = 4
    train_samples: int = 4
    eval_samples: int = 4
    eval_size: int = 20
    split_seed: int = 7
    max_turns: int = 40
    learning_rate: float = 1e-5
    minimum_donor_tasks: int = 16
    validation_only: bool = False
    # Optional fixed subset for a small end-to-end pilot, chosen before sampling.
    task_names: str | None = None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class RecordedDataset(SupervisedDataset):
    def __init__(self, paths: list[str], batch_size: int) -> None:
        self.paths = paths
        self.order = list(range(len(paths)))
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.paths) / self.batch_size)

    def set_epoch(self, seed: int = 0) -> None:
        self.order = list(range(len(self.paths)))
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        datums = []
        for position in self.order[index * self.batch_size : (index + 1) * self.batch_size]:
            row = cast(RecordedRollout, json.loads(Path(self.paths[position]).read_text()))
            if not eligible(row):
                raise ValueError("Ineligible donor in training manifest")
            total = sum(sum(d["weights"]) for d in row["datums"])
            if total <= 0:
                raise ValueError("No supervised action tokens")
            for datum in row["datums"]:
                inputs, targets, weights = (
                    datum["input_tokens"],
                    datum["target_tokens"],
                    datum["weights"],
                )
                if len(inputs) != len(targets) or len(inputs) != len(weights):
                    raise ValueError("Token/target/mask length mismatch")
                if inputs[1:] != targets[:-1]:
                    raise ValueError("Token targets are not shifted by one")
                if len(inputs) > 114688 or any(w not in (0, 1) for w in weights):
                    raise ValueError("Invalid context length or action mask")
                datums.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(inputs),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData.from_numpy(
                                np.array(targets, dtype=np.int64)
                            ),
                            "weights": tinker.TensorData.from_numpy(
                                np.array(weights, dtype=np.float32) / total
                            ),
                        },
                    )
                )
        return datums


@chz.chz
class RecordedDatasetBuilder(SupervisedDatasetBuilder):
    manifest: str
    batch_size: int = 4

    def __call__(self) -> tuple[SupervisedDataset, None]:
        content = json.loads(Path(self.manifest).read_text())
        paths = content["paths"]
        allowed = set(content["train_tasks"])
        seen: set[str] = set()
        for path in paths:
            row = json.loads(Path(path).read_text())
            task = row["task_name"]
            if task not in allowed or task in seen:
                raise ValueError("Heldout/duplicate task in donor manifest")
            seen.add(task)
        if not paths:
            raise ValueError("No donor trajectories")
        return RecordedDataset(paths, self.batch_size), None


def choose_donors(rows: list[tuple[Path, RecordedRollout]]) -> dict[str, list[str]]:
    """Use the same task support and one successful trajectory per task in both arms."""
    groups: dict[str, list[tuple[Path, RecordedRollout]]] = {}
    for path, row in rows:
        if eligible(row):
            groups.setdefault(row["task_name"], []).append((path, row))
    rng = random.Random(7)
    arms: dict[str, list[str]] = {"random_success": [], "short_success": []}
    for task in sorted(groups):
        candidates = sorted(groups[task], key=lambda item: item[1]["sample_index"])
        arms["random_success"].append(str(rng.choice(candidates)[0].resolve()))
        arms["short_success"].append(
            str(
                min(
                    candidates,
                    key=lambda item: (
                        item[1]["turns"],
                        item[1]["sampled_tokens"],
                        item[1]["sample_index"],
                    ),
                )[0].resolve()
            )
        )
    return arms


async def grade_patch(task: HarborTask, factory: SandboxFactory, patch: str | None) -> float:
    sandbox = await factory(task.task_dir / "environment", 3600)
    try:
        metadata = json.loads((task.task_dir / "metadata.json").read_text())
        baseline, merge = metadata["base_commit"], metadata["merge_commit"]
        check = await sandbox.run_command(
            "test ! -d /tests && test ! -d /solution && "
            f'test "$(git rev-parse HEAD)" = {shlex.quote(baseline)} && '
            'test "$(git rev-list --count HEAD)" = 1 && '
            f"! git cat-file -e {shlex.quote(merge)}",
            workdir="/workspace/repo",
            timeout=60,
        )
        if check.exit_code != 0:
            raise RuntimeError("Clean-room Git or hidden-material check failed")
        if patch is not None:
            await sandbox.write_file("/tmp/candidate.patch", patch)
            applied = await sandbox.run_command(
                "git apply --whitespace=nowarn /tmp/candidate.patch",
                workdir="/workspace/repo",
                timeout=60,
            )
            if applied.exit_code != 0:
                raise RuntimeError("Candidate patch failed to apply: " + applied.stderr[-500:])
        reward, _ = await HarborReward(task.task_dir / "tests", sandbox, 900, True)([])
        return reward
    finally:
        await sandbox.cleanup()


def delivery_metrics(results: list[TaskResult], directory: Path) -> dict[str, object]:
    total = len(results)
    passed = [r for r in results if r.error is None and r.reward == 1]
    tokens = []
    for r in results:
        path = directory / "rollouts" / f"{r.task_name}__{r.sample_index:02d}" / "trajectory.json"
        if path.exists():
            tokens.append(json.loads(path.read_text())["sampled_tokens"])
    return {
        "slots": total,
        "passed": len(passed),
        "infra_errors": sum(r.error is not None for r in results),
        "pass1": len(passed) / total,
        "mean_turns_all": sum(r.turns_used for r in results) / total,
        "mean_turns_success": sum(r.turns_used for r in passed) / len(passed) if passed else None,
        "sampled_tokens_recorded": sum(tokens),
        "token_slots_recorded": len(tokens),
        "sampled_tokens_per_delivered_success": sum(tokens) / len(passed) if passed else None,
        "accounting_scope": "final slots only; infrastructure retries excluded",
        # Per success includes failed trials, preventing cheap early failure from
        # looking like an efficiency win. Failed infra attempts are reported separately.
        "turns_per_delivered_success": sum(r.turns_used for r in results) / len(passed)
        if passed
        else None,
        "observed_success_ending_by_turn": {
            str(cap): sum(r.turns_used <= cap for r in passed) / total for cap in [20, 30, 40]
        },
    }


async def main(config: Config) -> None:
    load_env_file(Path(".env"))
    if config.max_turns != 40:
        raise ValueError("This experiment fixes the rollout budget at 40 turns")
    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "status.json"

    def state(stage: str, **details: object) -> None:
        write_json(state_path, {"stage": stage, **details})
        print(stage, json.dumps(details), flush=True)

    try:
        resolved = chz.asdict(config)
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != resolved:
            raise ValueError("Experiment config changed; use a fresh directory")
        write_json(config_path, resolved)
        snapshot = root / "tasks"
        if not snapshot.exists():
            staging = root / "tasks.pending"
            if staging.exists():
                shutil.rmtree(staging)
            with prepared_kokkos_tasks(Path(config.tasks_dir)) as prepared:
                staging.mkdir()
                for task in prepared:
                    shutil.copytree(task.task_dir, staging / task.task_name)
            staging.replace(snapshot)
        tasks = load_harbor_tasks_from_dir(snapshot)
        if config.task_names:
            wanted = set(config.task_names.split(","))
            tasks = [t for t in tasks if t.task_name in wanted]
            if len(tasks) != len(wanted):
                raise ValueError("Unknown task names")
        shuffled = list(tasks)
        random.Random(config.split_seed).shuffle(shuffled)
        heldout, training = shuffled[: config.eval_size], shuffled[config.eval_size :]
        if not heldout or not training:
            raise ValueError("Need disjoint nonempty train and heldout task sets")
        manifest = {
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "task_hashes": {t.task_name: _task_digest(t) for t in tasks},
            "train": [t.task_name for t in training],
            "heldout": [t.task_name for t in heldout],
        }
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Code or task snapshot changed; use a fresh experiment directory")
        write_json(manifest_path, manifest)
        if config.sandbox_backend == "contree":
            from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

            factory: SandboxFactory = ContreeDockerfileSandboxFactory(
                Path(config.cache_path),
                timeout=3600,
                allow_network=False,
            )
        elif config.sandbox_backend == "modal":
            factory = functools.partial(default_sandbox_factory, allow_network=False)
        else:
            raise ValueError("Unknown backend")
        semaphore = asyncio.Semaphore(config.concurrency)

        async def validate(task: HarborTask) -> None:
            path = root / "validation" / f"{task.task_name}.json"
            if path.exists() and json.loads(path.read_text()).get("passed"):
                return
            async with semaphore:
                nop = await grade_patch(task, factory, None)
                gold = (task.task_dir / "solution/gold.patch").read_text()
                oracle = await grade_patch(task, factory, gold)
                record = {
                    "task": task.task_name,
                    "nop": nop,
                    "oracle": oracle,
                    "passed": nop == 0 and oracle == 1,
                }
                write_json(path, record)
                print("validation", json.dumps(record), flush=True)
                if not record["passed"]:
                    raise RuntimeError("Oracle/NOP gate failed: " + task.task_name)

        state("validating_verifiers", tasks=len(tasks))
        # Await every bounded validation before advancing; one failure closes the gate.
        validation_results = await asyncio.gather(
            *(validate(t) for t in tasks), return_exceptions=True
        )
        failures = [str(r) for r in validation_results if isinstance(r, BaseException)]
        if failures:
            raise RuntimeError("Validation failed: " + "; ".join(failures[:10]))
        if config.validation_only:
            state("validation_complete")
            return

        async def evaluate(
            label: str, selected: list[HarborTask], samples: int, checkpoint: str | None = None
        ) -> list[TaskResult]:
            directory = root / label
            cfg = EvalConfig(
                model_name=config.model_name,
                output_path=str(directory),
                resume_dir=str(directory),
                max_turns=40,
                max_tokens=16384,
                temperature=1.0,
                thinking_effort=0.9,
                command_timeout=900,
                grader_timeout=900,
                max_tool_calls=80,
                max_trajectory_tokens=114688,
                max_sampled_tokens=65536,
                max_concurrency=config.concurrency,
                num_samples=samples,
                pass_at_k="1",
                sandbox_backend=config.sandbox_backend,
                allow_network=False,
                checkpoint_url=checkpoint,
                export_kokkos_rollouts=True,
            )
            result = await run_eval(cfg, selected, sandbox_factory=factory)
            write_json(directory / "efficiency.json", delivery_metrics(result, directory))
            if any(r.error for r in result):
                raise RuntimeError("Unresolved infrastructure errors in " + label)
            return result

        state(
            "baseline_and_collection",
            heldout_rollouts=len(heldout) * config.eval_samples,
            training_rollouts=len(training) * config.train_samples,
        )
        # Sequential stages keep the global sandbox concurrency bounded.
        await evaluate("baseline", heldout, config.eval_samples)
        await evaluate("collection", training, config.train_samples)
        state("qualifying_donors")
        by_name = {t.task_name: t for t in training}
        candidates: list[tuple[Path, RecordedRollout]] = []

        async def qualify(path: Path) -> None:
            row = cast(RecordedRollout, json.loads(path.read_text()))
            if not eligible(row) or row["task_name"] not in by_name:
                return
            patch = path.with_name("patch.diff").read_text()
            evidence = path.with_name("regrade.json")
            digest = hashlib.sha256(patch.encode()).hexdigest()
            if evidence.exists() and json.loads(evidence.read_text()).get("patch_sha256") == digest:
                reward = json.loads(evidence.read_text())["reward"]
            else:
                async with semaphore:
                    reward = await grade_patch(by_name[row["task_name"]], factory, patch)
                write_json(evidence, {"reward": reward, "patch_sha256": digest})
            if reward == 1:
                candidates.append((path, row))

        await asyncio.gather(
            *(qualify(p) for p in (root / "collection/rollouts").glob("*/trajectory.json"))
        )
        arms = choose_donors(candidates)
        n = len(arms["random_success"])
        if n < config.minimum_donor_tasks:
            raise RuntimeError(f"Only {n} qualified tasks; minimum is {config.minimum_donor_tasks}")
        write_json(
            root / "donor_summary.json",
            {
                "qualified_trajectories": len(candidates),
                "tasks": n,
                "different_donor_choices": sum(
                    a != b
                    for a, b in zip(arms["random_success"], arms["short_success"], strict=True)
                ),
            },
        )
        for arm, paths in arms.items():
            manifest = root / f"{arm}.json"
            write_json(manifest, {"paths": paths, "train_tasks": sorted(by_name)})
            builder = RecordedDatasetBuilder(manifest=str(manifest))
            dataset, _ = builder()
            # Validate every token mask and sequence before opening a training client.
            for i in range(len(dataset)):
                dataset.get_batch(i)
            state("training", arm=arm, donor_tasks=n, steps=len(dataset), max_turns=40)
            log_path = root / arm
            await train.main(
                train.Config(
                    log_path=str(log_path),
                    model_name=config.model_name,
                    recipe_name="kokkos_self_train",
                    dataset_builder=builder,
                    learning_rate=config.learning_rate,
                    lora_rank=32,
                    num_epochs=1,
                    lr_schedule="constant",
                    save_every=0,
                    eval_every=0,
                    wandb_project="kokkos-rl",
                    wandb_name=f"kokkos-rft40-{arm}-{root.name}",
                )
            )
            checkpoints = [
                json.loads(line)
                for line in (log_path / "checkpoints.jsonl").read_text().splitlines()
            ]
            checkpoint = checkpoints[-1]["sampler_path"]
            state("evaluating_checkpoint", arm=arm, checkpoint=checkpoint)
            await evaluate("eval_" + arm, heldout, config.eval_samples, checkpoint)
        state("complete", donor_tasks=n)
    except Exception as error:
        state("blocked", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
