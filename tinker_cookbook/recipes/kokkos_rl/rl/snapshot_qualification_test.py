import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl import snapshot_qualification as qualification
from tinker_cookbook.recipes.kokkos_rl.rl.validate_snapshot_test import task_with_metadata


def make_snapshot(root, monkeypatch):
    tasks = [
        task_with_metadata(root / "tasks", f"kokkos__kokkos-{number}")
        for number in (6375, 7074, 7089)
    ]
    hashes = {task.task_name: qualification._task_digest(task) for task in tasks}
    (root / "manifest.json").write_text(json.dumps({"task_hashes": hashes}))
    monkeypatch.setattr(qualification, "load_harbor_tasks_from_dir", lambda _: tasks)
    return tasks, hashes


def save_record(folder, task, digest, **overrides):
    record = {
        "task": task.task_name,
        "task_hash": digest,
        **asdict(qualification.policy_for_task(task)),
        "nop": 0,
        "oracle": 1,
        "passed": True,
        "stage": "complete",
        **overrides,
    }
    path = folder / f"{task.task_name}.json"
    path.write_text(json.dumps(record))
    return path


def test_qualification_rejects_wrong_payload_and_resources_preserves_failures(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    tasks, hashes = make_snapshot(snapshot, monkeypatch)
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    save_record(old, tasks[0], hashes[tasks[0].task_name])
    save_record(old, tasks[1], hashes[tasks[1].task_name], **asdict(qualification.Policy()))
    save_record(old, tasks[2], "old snapshot", passed=False, oracle=0)
    save_record(new, tasks[2], hashes[tasks[2].task_name], passed=False, oracle=0)
    config = qualification.Config(
        snapshot_dir=str(snapshot),
        evidence_dirs=(str(old), str(new)),
        output_path=str(tmp_path / "qualified.json"),
    )
    report = qualification.qualify(config)
    assert report["qualified"] == 1
    assert report["ready_for_sampling"] is False
    records = report["tasks"]
    assert records[tasks[1].task_name]["stage"] == "missing"
    assert len(records[tasks[1].task_name]["rejected_evidence"]) == 1
    assert records[tasks[2].task_name]["stage"] == "failed"
    assert len(records[tasks[2].task_name]["failed_evidence"]) == 1
    assert len(records[tasks[2].task_name]["rejected_evidence"]) == 1

    resolved = tmp_path / "resolved"
    resolved.mkdir()
    for task in tasks[1:]:
        save_record(resolved, task, hashes[task.task_name])
    complete = qualification.qualify(
        qualification.Config(
            snapshot_dir=str(snapshot),
            evidence_dirs=(str(old), str(new), str(resolved)),
            output_path=str(tmp_path / "resolved.json"),
        )
    )
    assert complete["ready_for_sampling"] is True
    assert complete["qualified"] == 3
    assert len(complete["tasks"][tasks[2].task_name]["failed_evidence"]) == 1
    assert complete["tasks"][tasks[0].task_name]["successful_evidence"][0]["sha256"]


def test_manifest_tampering_and_missing_evidence_directory_fail_closed(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    tasks, _ = make_snapshot(snapshot, monkeypatch)
    config = qualification.Config(
        snapshot_dir=str(snapshot),
        evidence_dirs=(str(tmp_path / "missing"),),
        output_path=str(tmp_path / "out.json"),
    )
    with pytest.raises(ValueError, match="Evidence directory does not exist"):
        qualification.qualify(config)
    (tasks[0].task_dir / "tests/test.sh").write_text("echo changed verifier\n")
    with pytest.raises(ValueError, match="exactly match"):
        qualification.qualify(config)
    assert not (tmp_path / "out.json").exists()


def test_independent_monitors_can_refresh_the_same_report(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    tasks, hashes = make_snapshot(snapshot, monkeypatch)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for task in tasks:
        save_record(evidence, task, hashes[task.task_name])
    output = tmp_path / "report.json"
    config = qualification.Config(
        snapshot_dir=str(snapshot), evidence_dirs=(str(evidence),), output_path=str(output)
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda _: qualification.qualify(config), range(20)))
    assert all(report["ready_for_sampling"] is True for report in reports)
    assert json.loads(output.read_text())["qualified"] == 3
    assert not list(tmp_path.glob(".report.json.*.tmp"))
