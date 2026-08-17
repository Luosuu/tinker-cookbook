from __future__ import annotations

import pytest

from tinker_cookbook.recipes.code_rl import code_grading
from tinker_cookbook.sandbox import SandboxBackend, SandboxResult


@pytest.mark.asyncio
async def test_contree_backend_dispatches_to_pool(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePool:
        async def run_in_workdir(
            self,
            files: dict[str, str],
            command: list[str],
            timeout: int,
        ) -> SandboxResult:
            captured.update(files=files, command=command, timeout=timeout)
            return SandboxResult(stdout="", stderr="", exit_code=0)

    monkeypatch.setattr(code_grading, "_get_contree_pool", lambda: FakePool())
    passed, details = await code_grading.sandbox_check_correctness(
        [{"input": "1\n", "output": "1\n", "metadata": {}}],
        "print(input())",
        backend=SandboxBackend.CONTREE,
    )

    assert passed
    assert details["exit_code"] == 0
    assert captured["command"] == ["python", "run.py"]
    assert set(captured["files"]) == {
        "test_cases.txt",
        "code.py",
        "testing_util.py",
        "run.py",
    }
