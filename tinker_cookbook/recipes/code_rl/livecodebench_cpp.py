from __future__ import annotations

import base64
import io
import json
import logging
import pickle
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import chz
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_url

from tinker_cookbook.recipes.code_rl.code_env import (
    DeepcoderEnvGroupBuilder,
)
from tinker_cookbook.recipes.code_rl.deepcoder_tool import DeepcoderTask
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxBackend

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/LiveCodeBench-CPP"
DATASET_REVISION = "6c21015d3095e5c45812950d683f5f3e65403c13"
DatasetSplit = Literal["v5_2408_2501", "v6_2408_2505"]

CPP_SYSTEM_MESSAGE = (
    "You are an expert C++17 competitive programmer. Solve the problem and return only the "
    "implementation inside a fenced ```cpp code block."
)


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Decode the dataset's pickled JSON string without allowing global imports."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(f"Global {module}.{name} is not allowed")


def decode_livecodebench_cpp_tests(value: object) -> list[dict[str, Any]]:
    """Decode plain JSON or the dataset's base64/zlib/pickle-wrapped JSON tests."""
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        raise TypeError(f"Expected encoded tests to be a string, got {type(value).__name__}")

    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError:
        compressed = base64.b64decode(value.encode("utf-8"), validate=True)
        pickled = zlib.decompress(compressed)
        decoded_json = _NoGlobalsUnpickler(io.BytesIO(pickled)).load()
        if isinstance(decoded_json, bytes):
            decoded_json = decoded_json.decode("utf-8")
        if not isinstance(decoded_json, str):
            raise TypeError(
                "Expected the compressed private tests to contain a JSON string, "
                f"got {type(decoded_json).__name__}"
            ) from None
        decoded = json.loads(decoded_json)

    if not isinstance(decoded, list):
        raise TypeError(f"Expected decoded tests to be a list, got {type(decoded).__name__}")

    tests: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, dict):
            raise TypeError(f"Expected each test to be a mapping, got {type(item).__name__}")
        testtype = str(item.get("testtype", "stdin"))
        if testtype not in {"stdin", "stdin_stdout", "functional"}:
            raise ValueError(f"Unsupported LiveCodeBench-CPP test type: {testtype}")
        tests.append(
            {
                "input": str(item.get("input", "")),
                "output": str(item.get("output", "")),
                "testtype": testtype,
                "metadata": {},
            }
        )
    return tests


def _build_cpp_prompt(row: Mapping[str, object]) -> str:
    question = row.get("question_content")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("LiveCodeBench-CPP row has no question_content")

    starter_code = row.get("starter_code")
    prompt = CPP_SYSTEM_MESSAGE + "\n\n" + question.strip()
    if isinstance(starter_code, str) and starter_code.strip():
        prompt += (
            "\n\nComplete the following starter code. Do not add a main function because the "
            "judge supplies one:\n```cpp\n" + starter_code.strip() + "\n```"
        )
    else:
        prompt += (
            "\n\nWrite a complete program that reads from stdin and writes to stdout. "
            "Do not hard-code the sample cases."
        )
    return prompt


def livecodebench_cpp_task_from_row(row: Mapping[str, object]) -> DeepcoderTask:
    public_tests = decode_livecodebench_cpp_tests(row.get("public_test_cases"))
    private_tests = decode_livecodebench_cpp_tests(row.get("private_test_cases"))
    tests = public_tests + private_tests
    if not tests:
        raise ValueError(f"LiveCodeBench-CPP row {row.get('question_id')} has no tests")

    starter_code = row.get("starter_code")
    return DeepcoderTask(
        problem=_build_cpp_prompt(row),
        tests=tests,
        starter_code=starter_code if isinstance(starter_code, str) and starter_code else None,
        language="cpp",
        dataset_name="livecodebench_cpp",
    )


def _load_livecodebench_cpp_split(
    split: DatasetSplit,
    revision: str,
) -> Dataset:
    logger.info("Loading %s split %s at revision %s", DATASET_NAME, split, revision)
    data_url = hf_hub_url(
        repo_id=DATASET_NAME,
        filename=f"test_{split}.jsonl",
        repo_type="dataset",
        revision=revision,
    )
    dataset = load_dataset(
        "json",
        data_files={split: data_url},
        split=split,
    )
    return cast(Dataset, dataset)


class LiveCodeBenchCppDataset(RLDataset):
    """Lazily decode large private tests only for the requested RL batch."""

    def __init__(
        self,
        rows: Dataset,
        batch_size: int,
        group_size: int,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        sandbox_backend: SandboxBackend | None,
        timeout: int,
        format_coef: float,
        max_generation_tokens: int | None,
        context_overflow_reward: float,
    ):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.sandbox_backend = sandbox_backend
        self.timeout = timeout
        self.format_coef = format_coef
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        builders: list[DeepcoderEnvGroupBuilder] = []
        for row_index in range(start, end):
            row = cast(dict[str, object], self.rows[row_index])
            builders.append(
                DeepcoderEnvGroupBuilder(
                    task=livecodebench_cpp_task_from_row(row),
                    model_name=self.model_name,
                    renderer_name=self.renderer_name,
                    max_turns=self.max_turns,
                    group_size=self.group_size,
                    sandbox_backend=self.sandbox_backend,
                    timeout=self.timeout,
                    format_coef=self.format_coef,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return builders

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size


@chz.chz
class LiveCodeBenchCppDatasetBuilder(RLDatasetBuilder):
    """Build mutually exclusive train/eval sets from LiveCodeBench-CPP."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    dataset_split: DatasetSplit = "v6_2408_2505"
    dataset_revision: str = DATASET_REVISION
    eval_size: int = 32
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    max_turns: int = 2
    format_coef: float = 0.1
    timeout: int = 30
    sandbox_backend: SandboxBackend | None = None
    seed: int = 0
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1

    def _make_dataset(self, rows: Dataset, group_size: int) -> LiveCodeBenchCppDataset:
        return LiveCodeBenchCppDataset(
            rows=rows,
            batch_size=self.batch_size,
            group_size=group_size,
            model_name=self.model_name_for_tokenizer,
            renderer_name=self.renderer_name,
            max_turns=self.max_turns,
            sandbox_backend=self.sandbox_backend,
            timeout=self.timeout,
            format_coef=self.format_coef,
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=self.context_overflow_reward,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        rows = _load_livecodebench_cpp_split(self.dataset_split, self.dataset_revision)
        if not 0 <= self.eval_size < len(rows):
            raise ValueError(f"eval_size must be in [0, {len(rows) - 1}], got {self.eval_size}")

        rows = rows.shuffle(seed=self.seed)
        eval_rows = rows.select(range(self.eval_size))
        train_rows = rows.select(range(self.eval_size, len(rows)))
        if self.max_train_examples is not None:
            train_rows = train_rows.select(range(min(self.max_train_examples, len(train_rows))))
        if self.max_eval_examples is not None:
            eval_rows = eval_rows.select(range(min(self.max_eval_examples, len(eval_rows))))

        train_dataset = self._make_dataset(train_rows, self.group_size)
        eval_dataset = self._make_dataset(eval_rows, 1) if len(eval_rows) else None
        logger.info(
            "Prepared LiveCodeBench-CPP: %d train, %d eval examples",
            len(train_rows),
            len(eval_rows),
        )
        return train_dataset, eval_dataset
