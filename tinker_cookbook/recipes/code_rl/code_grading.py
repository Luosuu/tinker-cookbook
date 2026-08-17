"""
Code grading utilities for RL training.

Supports two execution backends:
- sandboxfusion: Local Docker-based sandbox (default)
- modal: Cloud-based Modal sandbox
- contree: Cloud-based Nebius ConTree sandbox
"""

from __future__ import annotations

import json
import re
from typing import Any

from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL
from tinker_cookbook.sandbox import SandboxBackend, SandboxFusionClient

# Global sandbox backend clients (lazily initialized)
_sandboxfusion_client: SandboxFusionClient | None = None
_modal_pool: Any = None  # ModalSandboxPool, but avoid import at module level
_contree_pool: Any = None  # ContreeSandboxPool, but avoid import at module level
_cpp_modal_pool: Any = None  # ModalSandboxPool with a C++ toolchain
_cpp_contree_pool: Any = None  # ContreeSandboxPool with a C++ toolchain


CPP_GRADER_DRIVER = r"""
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

std::string trim(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool run_command(const std::string& command, const std::string& stage) {
    const int status = std::system(command.c_str());
    if (status == 0) {
        return true;
    }
    std::cerr << stage << " failed with status " << status << "\n";
    return false;
}

}  // namespace

int main() {
    const std::string mode = trim(read_file("mode.txt"));
    const int count = std::stoi(trim(read_file("count.txt")));
    const int timeout_seconds = std::stoi(trim(read_file("timeout.txt")));

    if (mode == "functional") {
        for (int index = 0; index < count; ++index) {
            const std::string suffix = std::to_string(index);
            const std::string compile =
                "g++ -std=c++17 -O2 -pipe solution_" + suffix + ".cpp -o solution_" +
                suffix + " 2> compile_" + suffix + ".stderr";
            if (!run_command(compile, "compile case " + suffix)) {
                std::cerr << read_file("compile_" + suffix + ".stderr");
                return 2;
            }
            const std::string execute =
                "timeout " + std::to_string(timeout_seconds) + "s ./solution_" + suffix +
                " > actual_" + suffix + ".txt 2> runtime_" + suffix + ".stderr";
            if (!run_command(execute, "functional case " + suffix)) {
                std::cerr << read_file("runtime_" + suffix + ".stderr");
                return 3;
            }
        }
        return 0;
    }

    if (mode != "stdin") {
        std::cerr << "unsupported test mode: " << mode << "\n";
        return 4;
    }

    if (!run_command(
            "g++ -std=c++17 -O2 -pipe solution.cpp -o solution 2> compile.stderr",
            "compile")) {
        std::cerr << read_file("compile.stderr");
        return 5;
    }
    for (int index = 0; index < count; ++index) {
        const std::string suffix = std::to_string(index);
        const std::string execute =
            "timeout " + std::to_string(timeout_seconds) + "s ./solution < input_" + suffix +
            ".txt > actual_" + suffix + ".txt 2> runtime_" + suffix + ".stderr";
        if (!run_command(execute, "stdin case " + suffix)) {
            std::cerr << read_file("runtime_" + suffix + ".stderr");
            return 6;
        }
        const std::string actual = trim(read_file("actual_" + suffix + ".txt"));
        const std::string expected = trim(read_file("expected_" + suffix + ".txt"));
        if (actual != expected) {
            std::cerr << "wrong answer on case " << suffix << "\n";
            std::cerr << "expected: " << expected.substr(0, 500) << "\n";
            std::cerr << "actual: " << actual.substr(0, 500) << "\n";
            return 7;
        }
    }
    return 0;
}
"""


def _get_sandboxfusion_client() -> SandboxFusionClient:
    """Get or create the SandboxFusion client."""
    global _sandboxfusion_client
    if _sandboxfusion_client is None:
        _sandboxfusion_client = SandboxFusionClient()
    return _sandboxfusion_client


def _get_modal_pool():
    """Get or create the Modal sandbox pool."""
    global _modal_pool
    if _modal_pool is None:
        import modal

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandboxPool

        image = modal.Image.debian_slim().pip_install("numpy")
        _modal_pool = ModalSandboxPool(image=image)
    return _modal_pool


def _get_contree_pool():
    """Get or create the ConTree sandbox pool."""
    global _contree_pool
    if _contree_pool is None:
        from tinker_cookbook.sandbox.contree_sandbox import ContreeSandboxPool

        _contree_pool = ContreeSandboxPool(
            image="python:3.12-slim",
            setup_command="python -m pip install --no-cache-dir numpy",
        )
    return _contree_pool


def _get_cpp_modal_pool():
    """Get or create a Modal pool whose image contains g++."""
    global _cpp_modal_pool
    if _cpp_modal_pool is None:
        import modal

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandboxPool

        image = modal.Image.debian_slim().apt_install("g++")
        _cpp_modal_pool = ModalSandboxPool(image=image)
    return _cpp_modal_pool


def _get_cpp_contree_pool():
    """Get or create a ConTree pool whose image contains g++."""
    global _cpp_contree_pool
    if _cpp_contree_pool is None:
        from tinker_cookbook.sandbox.contree_sandbox import ContreeSandboxPool

        _cpp_contree_pool = ContreeSandboxPool(image="gcc:14-bookworm")
    return _cpp_contree_pool


async def close_cpp_sandbox_pools() -> None:
    """Terminate shared C++ cloud pools after an evaluation or training run."""
    global _cpp_modal_pool, _cpp_contree_pool
    pools = [pool for pool in (_cpp_modal_pool, _cpp_contree_pool) if pool is not None]
    _cpp_modal_pool = None
    _cpp_contree_pool = None
    for pool in pools:
        await pool.terminate()


def extract_code_from_model(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    """Convert test cases to LiveCodeBench format for the test runner."""
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]

    sample_dict: dict[str, Any] = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is None:
            raise AssertionError(f"Function name missing in metadata: {metadata}. Sample: {sample}")
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


async def _check_with_sandboxfusion(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using SandboxFusion backend."""
    client = _get_sandboxfusion_client()

    return await client.run(
        code=TEST_CODE % {"timeout": timeout},
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
        },
        timeout=total_timeout,
    )


async def _check_with_modal(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using Modal sandbox."""
    pool = _get_modal_pool()
    result = await pool.run_in_workdir(
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
            "run.py": TEST_CODE % {"timeout": timeout},
        },
        command=["python", "run.py"],
        timeout=total_timeout,
    )
    return result.exit_code == 0, {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def _check_with_contree(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using a ConTree sandbox."""
    pool = _get_contree_pool()
    result = await pool.run_in_workdir(
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
            "run.py": TEST_CODE % {"timeout": timeout},
        },
        command=["python", "run.py"],
        timeout=total_timeout,
    )
    return result.exit_code == 0, {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _cpp_grader_files(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int,
) -> dict[str, str]:
    """Build files consumed by the C++ grader driver."""
    test_types = {str(test.get("testtype", "stdin")) for test in sample}
    functional = test_types == {"functional"}
    stdin_types = {"stdin", "stdin_stdout"}
    if not functional and not test_types.issubset(stdin_types):
        raise ValueError(f"Mixed or unsupported C++ test types: {sorted(test_types)}")

    files = {
        "mode.txt": "functional" if functional else "stdin",
        "count.txt": str(len(sample)),
        "timeout.txt": str(timeout),
    }
    if functional:
        prelude = "#include <bits/stdc++.h>\nusing namespace std;\n"
        for index, test in enumerate(sample):
            files[f"solution_{index}.cpp"] = prelude + generation + "\n" + str(test["input"])
    else:
        files["solution.cpp"] = generation
        for index, test in enumerate(sample):
            files[f"input_{index}.txt"] = str(test.get("input", ""))
            files[f"expected_{index}.txt"] = str(test.get("output", ""))
    return files


async def _check_cpp_with_sandboxfusion(
    files: dict[str, str],
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    client = _get_sandboxfusion_client()
    return await client.run(
        code=CPP_GRADER_DRIVER,
        files=files,
        timeout=total_timeout,
        language="cpp",
    )


async def _check_cpp_with_pool(
    files: dict[str, str],
    total_timeout: int,
    backend: SandboxBackend,
) -> tuple[bool, dict[str, Any]]:
    pool = _get_cpp_modal_pool() if backend == SandboxBackend.MODAL else _get_cpp_contree_pool()
    result = await pool.run_in_workdir(
        files={**files, "grader.cpp": CPP_GRADER_DRIVER},
        command=[
            "sh",
            "-c",
            "g++ -std=c++17 -O2 grader.cpp -o grader && ./grader",
        ],
        timeout=total_timeout,
    )
    return result.exit_code == 0, {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def sandbox_check_cpp_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 30,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Compile and grade a C++17 submission in a sandbox."""
    if not sample:
        raise ValueError("Sample must contain at least one C++ test case")

    use_backend = backend or SandboxBackend.SANDBOXFUSION
    try:
        files = _cpp_grader_files(sample, generation, timeout)
        functional = files["mode.txt"] == "functional"
        compile_budget = 30 * len(sample) if functional else 30
        total_timeout = compile_budget + (timeout + 2) * len(sample) + 10

        if use_backend == SandboxBackend.SANDBOXFUSION:
            return await _check_cpp_with_sandboxfusion(files, total_timeout)
        if use_backend in {SandboxBackend.MODAL, SandboxBackend.CONTREE}:
            return await _check_cpp_with_pool(files, total_timeout, use_backend)
        raise ValueError(f"Invalid sandbox backend: {use_backend}")
    except Exception as exc:
        return False, {"error": str(exc)}


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code using sandbox execution.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds
        backend: Sandbox backend to use (defaults to "sandboxfusion")

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)
    use_backend = backend or SandboxBackend.SANDBOXFUSION

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        if use_backend == SandboxBackend.MODAL:
            return await _check_with_modal(test_cases, generation, timeout, total_timeout)
        elif use_backend == SandboxBackend.CONTREE:
            return await _check_with_contree(test_cases, generation, timeout, total_timeout)
        elif use_backend == SandboxBackend.SANDBOXFUSION:
            return await _check_with_sandboxfusion(test_cases, generation, timeout, total_timeout)
        else:
            raise ValueError(f"Invalid sandbox backend: {use_backend}")

    except Exception as e:
        return False, {"error": str(e)}


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert TACO-style tests to LiveCodeBench format."""
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    n = max(len(inputs), len(outputs))

    test_cases: list[dict[str, Any]] = []
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        if isinstance(out, list):
            out = out[0] if out else ""
        case: dict[str, Any] = {
            "input": inp,
            "output": out,
            "metadata": {},
        }
        if "fn_name" in tests:
            case["testtype"] = "functional"
            case["metadata"]["func_name"] = tests["fn_name"]
        else:
            case["testtype"] = "stdin_stdout"
        test_cases.append(case)

    return test_cases
