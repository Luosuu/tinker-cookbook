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


def test_cpp_grader_files_for_stdin_tests() -> None:
    files = code_grading._cpp_grader_files(
        [
            {"input": "1\n", "output": "1\n", "testtype": "stdin"},
            {"input": "2\n", "output": "2\n", "testtype": "stdin"},
        ],
        "#include <iostream>\nint main() { int x; std::cin >> x; std::cout << x; }",
        timeout=7,
    )

    assert files["mode.txt"] == "stdin"
    assert files["count.txt"] == "2"
    assert files["timeout.txt"] == "7"
    assert files["input_1.txt"] == "2\n"
    assert files["expected_1.txt"] == "2\n"


def test_cpp_grader_files_for_functional_tests() -> None:
    files = code_grading._cpp_grader_files(
        [
            {
                "input": "int main() { return Solution().answer() == 42 ? 0 : 1; }",
                "testtype": "functional",
            }
        ],
        "class Solution { public: int answer() { return 42; } };",
        timeout=30,
    )

    assert files["mode.txt"] == "functional"
    assert "#include <bits/stdc++.h>" in files["solution_0.cpp"]
    assert "class Solution" in files["solution_0.cpp"]
    assert "int main()" in files["solution_0.cpp"]


@pytest.mark.asyncio
async def test_cpp_contree_backend_dispatches_to_cpp_pool(monkeypatch) -> None:
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

    monkeypatch.setattr(code_grading, "_get_cpp_contree_pool", lambda: FakePool())
    passed, details = await code_grading.sandbox_check_cpp_correctness(
        [{"input": "1\n", "output": "1\n", "testtype": "stdin"}],
        "#include <iostream>\nint main() { int x; std::cin >> x; std::cout << x; }",
        backend=SandboxBackend.CONTREE,
    )

    assert passed
    assert details["exit_code"] == 0
    assert captured["command"] == [
        "sh",
        "-c",
        "g++ -std=c++17 -O2 grader.cpp -o grader && ./grader",
    ]
    assert "grader.cpp" in captured["files"]
