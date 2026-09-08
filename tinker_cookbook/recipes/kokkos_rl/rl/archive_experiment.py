"""Archive an existing experiment to W&B and import its conversations into Weave.

This performs no inference or training. Raw files remain byte-exact in an Artifact;
Weave contains the recorded conversations, not reconstructed execution timings.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import uuid
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(plan_path: Path, output: Path) -> None:
    plan = read(plan_path)
    repository = Path(plan["repository"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        raise ValueError("Use the existing prepared archive or a new output directory")
    files = {}
    for relative in plan["include"]:
        root = repository / relative
        for path in [root] if root.is_file() else sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(repository):
                raise ValueError("Archive sources must be regular repository files")
            if path.name == ".env" or path.name.startswith(".env."):
                raise ValueError("Credential file in archive scope")
            files[str(path.relative_to(repository))] = path
    # Detect actual locally configured credentials without printing their values.
    secrets = [
        v.encode()
        for k, v in os.environ.items()
        if any(word in k for word in ("API_KEY", "SECRET", "ACCESS_TOKEN")) and len(v) >= 16
    ]
    manifest = {}
    first_by_hash = {}
    with tarfile.open(output / "experiment.tar.gz", "w:gz", compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            raw = path.read_bytes()
            if any(secret in raw for secret in secrets):
                raise ValueError(f"Configured credential detected in {name}; nothing uploaded")
            sha = hashlib.sha256(raw).hexdigest()
            manifest[name] = {"sha256": sha, "bytes": len(raw)}
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            if sha in first_by_hash:
                info.type = tarfile.LNKTYPE
                info.linkname = first_by_hash[sha]
                archive.addfile(info)
            else:
                first_by_hash[sha] = name
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
    trajectories = []
    seen = set()
    for source in plan["result_sources"]:
        folder = repository / source["directory"]
        for line in (folder / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (source["split"], source["arm"], row["task_name"], row["sample_index"])
            if key in seen:
                raise ValueError("Duplicate canonical trajectory")
            seen.add(key)
            slot = f"{row['task_name']}__{row['sample_index']:02d}"
            rollout = folder / "rollouts" / slot
            messages = rollout / "messages.json"
            metadata = {
                n: read(rollout / n)
                for n in ("candidate.json", "failure.json", "verifier.json")
                if (rollout / n).exists()
            }
            patch = rollout / "candidate.patch"
            if not patch.exists():
                patch = rollout / "patch.diff"
            trace = (
                read(rollout / "trajectory.json") if (rollout / "trajectory.json").exists() else {}
            )
            task_instruction = repository / plan["tasks"] / row["task_name"] / "instruction.md"
            trajectories.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            plan["experiment_id"] + "/" + "/".join(map(str, key)),
                        )
                    ),
                    "split": source["split"],
                    "arm": source["arm"],
                    "task": row["task_name"],
                    "sample": row["sample_index"],
                    "checkpoint": source["checkpoint"],
                    "source": str(rollout.relative_to(repository)),
                    "result": row,
                    "logical_score": row["reward"] if row.get("error") is None else None,
                    "messages": read(messages) if messages.exists() else None,
                    "instruction": task_instruction.read_text()
                    if task_instruction.exists()
                    else None,
                    "patch": patch.read_text() if patch.exists() else None,
                    "metadata": metadata,
                    "stop_reason": trace.get("stop_reason"),
                    "sampled_tokens": trace.get("sampled_tokens"),
                    "audit_flags": trace.get("audit_flags", []),
                    "raw_files": [
                        str(f.relative_to(repository))
                        for f in sorted(rollout.rglob("*"))
                        if f.is_file()
                    ],
                }
            )
    if len(trajectories) != plan["expected_trajectories"]:
        raise ValueError("Canonical trajectory count mismatch")
    write(output / "file_manifest.json", manifest)
    write(output / "trajectories.json", trajectories)
    write(output / "plan.json", plan)
    write(
        output / "prepared.json",
        {
            "file_count": len(manifest),
            "unique_file_contents": len(first_by_hash),
            "source_bytes": sum(v["bytes"] for v in manifest.values()),
            "trajectories": len(trajectories),
            "missing_message_files": sum(t["messages"] is None for t in trajectories),
            "files": {
                name: checksum(output / name)
                for name in (
                    "experiment.tar.gz",
                    "file_manifest.json",
                    "trajectories.json",
                    "plan.json",
                )
            },
        },
    )


def upload(output: Path) -> None:
    import wandb
    import weave

    prepared, plan = read(output / "prepared.json"), read(output / "plan.json")
    for name, sha in prepared["files"].items():
        if checksum(output / name) != sha:
            raise ValueError("Prepared archive changed")
    state_path = output / "upload_state.json"
    state = read(state_path) if state_path.exists() else {}
    if state.get("complete"):
        print(json.dumps(state), flush=True)
        return
    entity, project = plan["project"].split("/")
    run = wandb.init(
        entity=entity,
        project=project,
        id=plan["run_id"],
        resume="allow",
        name=plan["experiment_id"],
        job_type="historical-archive",
        config=plan["run_config"],
        dir=str(output),
        save_code=False,
    )
    assert run is not None
    if "artifact" not in state:
        artifact = wandb.Artifact(
            plan["experiment_id"], type="experiment-archive", metadata=prepared
        )
        for name in (*prepared["files"], "prepared.json"):
            artifact.add_file(str(output / name), name=name)
        logged = run.log_artifact(artifact)
        logged.wait()
        state.update(artifact=f"{entity}/{project}/{logged.name}", run_url=run.url)
        write(state_path, state)
    for key, value in plan["metrics"].items():
        run.summary[key] = value
    run.summary["raw_archive"] = state["artifact"]
    run.summary["checkpoint_registry"] = plan["checkpoints"]
    run.summary["historical_import"] = True
    client = weave.init(plan["project"], settings={"print_call_link": False})
    trajectories = read(output / "trajectories.json")
    existing = {
        call.id
        for call in client.get_calls(columns=["id", "ended_at"])
        if call.ended_at is not None
    }
    for index, t in enumerate(trajectories):
        if t["id"] in existing:
            continue
        attrs = {k: t[k] for k in ("split", "arm", "task", "sample", "checkpoint", "source")}
        attrs.update(
            experiment_id=plan["experiment_id"],
            historical_import=True,
            raw_artifact=state["artifact"],
            timing_semantics="import time, not original execution",
        )
        call = client.create_call(
            "archived_rl_trajectory",
            inputs={
                "instruction": t["instruction"],
                "messages": t["messages"],
                "missing_original_messages": t["messages"] is None,
                "raw_files": t["raw_files"],
            },
            attributes=attrs,
            display_name=f"{t['split']}/{t['arm']}/{t['task']}/{t['sample']}",
            use_stack=False,
            _call_id_override=t["id"],
        )
        client.finish_call(
            call,
            output={
                k: t[k]
                for k in (
                    "result",
                    "logical_score",
                    "patch",
                    "metadata",
                    "stop_reason",
                    "sampled_tokens",
                    "audit_flags",
                )
            },
        )
        if (index + 1) % 20 == 0:
            client.flush()
            print(
                json.dumps({"imported_through": index + 1, "total": len(trajectories)}), flush=True
            )
    client.flush()
    found = {
        call.id
        for call in client.get_calls(columns=["id", "ended_at"])
        if call.ended_at is not None
    }
    expected = {t["id"] for t in trajectories}
    if not expected.issubset(found):
        raise RuntimeError(f"Remote trace verification incomplete: {len(expected - found)} missing")
    remote = wandb.Api().artifact(state["artifact"])
    if set(remote.manifest.entries) != {*prepared["files"], "prepared.json"}:
        raise RuntimeError("Remote Artifact file manifest mismatch")
    state.update(
        complete=True,
        verified_trajectories=len(expected),
        weave_url=f"https://wandb.ai/{plan['project']}/weave/calls",
        artifact_version=remote.version,
    )
    write(state_path, state)
    run.summary["verified_trajectories"] = len(expected)
    run.summary["weave_url"] = state["weave_url"]
    run.finish()
    print(json.dumps(state), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "upload"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file

    load_env_file(args.env_file)
    if args.mode == "prepare":
        if args.plan is None:
            parser.error("prepare requires --plan")
        prepare(args.plan, args.output.resolve())
    else:
        upload(args.output.resolve())


if __name__ == "__main__":
    main()
