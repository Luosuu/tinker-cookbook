import json
import tarfile

import pytest

from tinker_cookbook.recipes.kokkos_rl.rl.archive_experiment import prepare


def fixture(tmp_path, error=None):
    repo = tmp_path / "repo"
    rollout = repo / "phase/rollouts/task__00"
    rollout.mkdir(parents=True)
    message = [{"role": "assistant", "content": "complete " * 10000}]
    (rollout / "messages.json").write_text(json.dumps(message))
    result = {"task_name": "task", "sample_index": 0, "reward": 0, "error": error}
    (repo / "phase/results.jsonl").write_text(json.dumps(result) + "\n")
    plan = {
        "repository": str(repo),
        "include": ["phase"],
        "experiment_id": "fixture",
        "result_sources": [
            {"directory": "phase", "split": "heldout", "arm": "a", "checkpoint": "fixed"}
        ],
        "tasks": "tasks",
        "expected_trajectories": 1,
    }
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return p, tmp_path / "output", repo, message


def test_archive_keeps_full_conversation_and_infra_missing(tmp_path):
    plan, output, repo, messages = fixture(tmp_path, error="sandbox unavailable")
    prepare(plan, output)
    row = json.loads((output / "trajectories.json").read_text())[0]
    assert row["messages"] == messages
    assert row["logical_score"] is None
    with tarfile.open(output / "experiment.tar.gz") as archive:
        member = archive.extractfile("phase/rollouts/task__00/messages.json")
        assert member is not None
        assert member.read() == (repo / "phase/rollouts/task__00/messages.json").read_bytes()


def test_real_model_failure_stays_zero(tmp_path):
    plan, output, _, _ = fixture(tmp_path)
    prepare(plan, output)
    assert json.loads((output / "trajectories.json").read_text())[0]["logical_score"] == 0


def test_duplicate_canonical_trajectory_rejected(tmp_path):
    plan, output, _, _ = fixture(tmp_path)
    value = json.loads(plan.read_text())
    value["result_sources"] *= 2
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(plan, output)
    assert not (output / "prepared.json").exists()


def test_actual_credential_blocks_preparation(tmp_path, monkeypatch):
    plan, output, repo, _ = fixture(tmp_path)
    secret = "fixture-credential-never-upload-12345"
    monkeypatch.setenv("EXAMPLE_API_KEY", secret)
    (repo / "phase/unsafe.log").write_text(secret)
    with pytest.raises(ValueError, match="credential"):
        prepare(plan, output)
    assert not (output / "prepared.json").exists()


def test_missing_messages_remain_missing(tmp_path):
    plan, output, repo, _ = fixture(tmp_path)
    (repo / "phase/rollouts/task__00/messages.json").unlink()
    prepare(plan, output)
    assert json.loads((output / "trajectories.json").read_text())[0]["messages"] is None


def test_deduplicated_tar_restores_equal_files(tmp_path):
    plan, output, repo, _ = fixture(tmp_path)
    (repo / "phase/a").write_bytes(b"identical")
    (repo / "phase/b").write_bytes(b"identical")
    prepare(plan, output)
    with tarfile.open(output / "experiment.tar.gz") as archive:
        assert archive.getmember("phase/b").islnk()
        assert archive.extractfile("phase/b").read() == b"identical"
