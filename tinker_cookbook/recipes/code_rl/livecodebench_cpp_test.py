from __future__ import annotations

import base64
import json
import pickle
import zlib

import pytest
from datasets import Dataset

from tinker_cookbook.recipes.code_rl import livecodebench_cpp
from tinker_cookbook.recipes.code_rl.livecodebench_cpp import (
    LiveCodeBenchCppDatasetBuilder,
    decode_livecodebench_cpp_tests,
    livecodebench_cpp_task_from_row,
)


def _encode_private_tests(tests: list[dict[str, str]]) -> str:
    payload = pickle.dumps(json.dumps(tests))
    return base64.b64encode(zlib.compress(payload)).decode("utf-8")


def _row(index: int, *, functional: bool = False) -> dict[str, str]:
    testtype = "functional" if functional else "stdin"
    test_input = "int main() { return 0; }" if functional else f"{index}\n"
    return {
        "question_content": f"Problem {index}",
        "question_id": str(index),
        "starter_code": "class Solution {};" if functional else "",
        "public_test_cases": "",
        "private_test_cases": _encode_private_tests(
            [{"input": test_input, "output": "", "testtype": testtype}]
        ),
    }


def test_decode_compressed_private_tests() -> None:
    tests = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]

    assert decode_livecodebench_cpp_tests(_encode_private_tests(tests)) == [
        {"input": "1\n", "output": "1\n", "testtype": "stdin", "metadata": {}}
    ]


def test_builds_cpp_task_for_functional_problem() -> None:
    task = livecodebench_cpp_task_from_row(_row(1, functional=True))

    assert task.language == "cpp"
    assert task.dataset_name == "livecodebench_cpp"
    assert task.tests[0]["testtype"] == "functional"
    assert "Do not add a main function" in task.problem
    assert "class Solution" in task.problem


@pytest.mark.asyncio
async def test_builder_creates_disjoint_train_and_eval_sets(monkeypatch) -> None:
    rows = Dataset.from_list([_row(index) for index in range(5)])
    monkeypatch.setattr(
        livecodebench_cpp,
        "_load_livecodebench_cpp_split",
        lambda split, revision: rows,
    )
    builder = LiveCodeBenchCppDatasetBuilder(
        model_name_for_tokenizer="test-model",
        batch_size=2,
        group_size=3,
        eval_size=2,
        seed=7,
    )

    train, evaluation = await builder()

    assert len(train) == 2
    assert evaluation is not None
    assert len(evaluation) == 1
    train_prompts = {
        env_builder.task.problem
        for batch_index in range(len(train))
        for env_builder in train.get_batch(batch_index)
    }
    eval_prompts = {
        env_builder.task.problem
        for batch_index in range(len(evaluation))
        for env_builder in evaluation.get_batch(batch_index)
    }
    assert len(train_prompts) == 3
    assert len(eval_prompts) == 2
    assert train_prompts.isdisjoint(eval_prompts)
