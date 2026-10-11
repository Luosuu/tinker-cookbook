from __future__ import annotations

import json
from pathlib import Path

from notes.pi_harness.taxonomy import load_block, report

PATCH = "diff --git a/core/src/Kokkos_View.hpp b/core/src/Kokkos_View.hpp\n"
JUNK = "diff --git a/workspace/bash_events/1.json b/workspace/bash_events/1.json\n"


def _episode(
    block: Path,
    run: str,
    task: str,
    *,
    reward: float,
    patch: str,
    checks: list,
    commands: list[str],
    native: bool = False,
    error: str | None = None,
) -> None:
    d = block / "runs" / run / "results" / task
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {"task_name": task, "reward": reward, "error": error, "turns_used": len(commands)}
        )
    )
    (d / "candidate.patch").write_text(patch)
    grading = {"diagnostics": {"checks": checks}}
    if native:
        (d / "grading_logs").mkdir()
        (d / "grading_logs" / f"{task}.json").write_text(json.dumps(grading))
        turns = [{"function_calls": [{"arguments": json.dumps({"command": c})}]} for c in commands]
        (d / f"{task}.json").write_text(json.dumps(turns))
    else:
        (d / "grading.json").write_text(json.dumps(grading))
        (d / "events.jsonl").write_text(
            "".join(json.dumps({"type": "bash", "command": c}) + "\n" for c in commands)
        )


def test_outcomes_read_each_arms_own_evidence(tmp_path: Path) -> None:
    build_fail = [{"id": "build", "stage": "build", "status": "failed"}]
    f2p_fail = [
        {"id": "build", "stage": "build", "status": "passed"},
        {"id": "f2p-0", "stage": "test", "status": "failed"},
    ]
    _episode(
        tmp_path,
        "native-d0",
        "t1",
        reward=0.0,
        patch=JUNK,
        checks=build_fail,
        commands=["cat a"] * 40,
        native=True,
    )
    _episode(
        tmp_path,
        "native-d1",
        "t1",
        reward=1.0,
        patch=PATCH,
        checks=[],
        commands=["sed -i s/a/b/ x", "cmake --build build"],
        native=True,
    )
    _episode(
        tmp_path,
        "pi-d0",
        "t1",
        reward=0.0,
        patch=PATCH,
        checks=build_fail,
        commands=["cat a", "sed -i s/a/b/ x"],
    )
    _episode(
        tmp_path,
        "pi-d1",
        "t1",
        reward=0.0,
        patch=PATCH,
        checks=f2p_fail,
        commands=["sed -i s/a/b/ x", "ninja X"],
    )
    _episode(
        tmp_path,
        "pi-d2",
        "t1",
        reward=0.0,
        patch="",
        checks=[],
        commands=[],
        error="RuntimeError: 502",
    )
    # A harness build check after the edit is not the model building its change.
    events = tmp_path / "runs/pi-d0/results/t1/events.jsonl"
    probe = {"type": "bash", "command": "cmake --build build --target X", "internal": True}
    events.write_text(events.read_text() + json.dumps(probe) + "\n")
    episodes = load_block(tmp_path)
    assert [c for e in episodes if e.arm == "pi" and e.draw == "0" for c in e.commands] == [
        "cat a",
        "sed -i s/a/b/ x",
    ]
    out = report(episodes)
    native, pi = out["arms"]["native"], out["arms"]["pi"]
    assert native["failure_outcomes"] == {"no_source_change": 1}, "sandbox files are not edits"
    assert native["failed_at_request_cap"] == 1
    assert pi["failure_outcomes"] == {"build_failed": 1, "f2p_failed": 1}
    assert pi["valid"] == 2 and pi["failed_with_unbuilt_final_edit"] == 1
    assert out["tasks"] == {"mixed": 1}
