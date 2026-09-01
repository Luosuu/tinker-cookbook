"""Audit Kokkos benchmark problem statements with an effort-conditioned Inkling model.

The auditor compares each public problem statement with the private verifier patches, but asks
the model to emit only the observable contract needed by a coding agent. Reports are resumable
and can be reviewed before ``--apply`` updates the source JSONL and exported Harbor tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import export_instance
from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance
from tinker_cookbook.renderers import Message, ParseTermination, get_renderer, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

SYSTEM_PROMPT = """You audit repository-level coding benchmark instructions for fairness.
Return exactly one JSON object and no surrounding prose, with this schema:
{
  "instance_id": "unchanged input id",
  "assessment": "clear" | "needs_revision" | "invalid",
  "issues": ["short, concrete mismatch or ambiguity"],
  "revised_problem_statement": "complete replacement statement",
  "confidence": 0.0
}

Compare the public statement with the private test and reference patches. A fair statement must
name any required public API, compatibility mode, observable behavior, and important edge cases
that the verifier requires but a competent developer could not infer from the repository. Correct
typos in required API names. Resolve mutually exclusive alternatives. Do not require an exact
implementation when multiple implementations satisfy the behavior.

The replacement is shown to the coding agent, so never reveal private test names, test commands,
patch hunks, reference implementation details, exact diagnostic strings unless the wording itself
is the public contract, or that private patches were supplied. Do not tell the agent to add or edit
tests, build files, or CI. Keep clear statements unchanged. For revisions, preserve useful context
from the original while writing a concise standalone issue statement. Mark a task invalid when its
essential request is test/build/CI-only or cannot be made fair without changing the verifier."""

REFINEMENT_SYSTEM_PROMPT = """You edit a first-pass benchmark instruction audit into a fair,
implementation-neutral coding task. Return exactly one JSON object using this schema:
{
  "instance_id": "unchanged input id",
  "assessment": "clear" | "needs_revision" | "invalid",
  "issues": ["short reason the original needed revision"],
  "revised_problem_statement": "complete replacement statement",
  "confidence": 0.0
}

The first-pass draft was produced while viewing private verifier patches and may overfit them. Keep
only requirements a developer needs to understand the public contract: the affected public API,
the observed bug, required behavior, important edge cases, and compatibility modes. Remove private
test identities, instructions to add/edit tests, CMake or CI work, exact file paths, patch structure,
reference-only internal helper names, prescribed algorithms, and exact diagnostic prose. Do not
mention private evidence or upstream links. It is acceptable to name an internal type only when
that type is itself the API the task asks to add or change. Use concise prose of at most 300 words;
do not turn the reference patch into a checklist. Mark the task invalid if its essential request is
only tests, build configuration, or CI and no production behavior remains. Preserve the assessment
"clear" and the original text when no revision is needed."""


@dataclass(frozen=True)
class AuditResult:
    instance_id: str
    assessment: str
    issues: tuple[str, ...]
    revised_problem_statement: str
    confidence: float
    termination: str

    @classmethod
    def from_model_text(
        cls, text: str, *, expected_instance_id: str, termination: ParseTermination
    ) -> AuditResult:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("model response did not contain a JSON object")
        value = json.loads(match.group(0))
        instance_id = str(value["instance_id"])
        if instance_id != expected_instance_id:
            raise ValueError(f"expected instance_id {expected_instance_id}, got {instance_id}")
        assessment = str(value["assessment"])
        if assessment not in {"clear", "needs_revision", "invalid"}:
            raise ValueError(f"unsupported assessment: {assessment}")
        statement = str(value["revised_problem_statement"]).strip()
        if not statement or len(statement) > 6000:
            raise ValueError("revised_problem_statement must contain 1-6000 characters")
        confidence = float(value["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        issues = tuple(str(item).strip() for item in value.get("issues", []) if str(item).strip())
        return cls(
            instance_id=instance_id,
            assessment=assessment,
            issues=issues,
            revised_problem_statement=statement,
            confidence=confidence,
            termination=termination.value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "assessment": self.assessment,
            "issues": list(self.issues),
            "revised_problem_statement": self.revised_problem_statement,
            "confidence": self.confidence,
            "termination": self.termination,
        }


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _read_instances(path: Path) -> list[KokkosInstance]:
    return [
        KokkosInstance.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _prompt(instance: KokkosInstance) -> str:
    return f"""Audit this benchmark task.

INSTANCE ID: {instance.instance_id}
TITLE: {instance.title}

PUBLIC PROBLEM STATEMENT:
<problem_statement>
{instance.problem_statement.strip()}
</problem_statement>

PRIVATE TEST PATCH (evidence only; do not quote or identify its tests):
<test_patch>
{instance.test_patch.strip()}
</test_patch>

PRIVATE REFERENCE PATCH (evidence only; do not disclose its implementation):
<reference_patch>
{instance.code_patch.strip()}
</reference_patch>
"""


async def _audit_one(
    instance: KokkosInstance,
    *,
    sampling_client: tinker.SamplingClient,
    renderer: TmlV0Renderer,
    effort: float,
    max_tokens: int,
    attempts: int,
) -> AuditResult:
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=_prompt(instance)),
    ]
    last_error: Exception | None = None
    for _attempt in range(attempts):
        prompt = renderer.build_generation_prompt(messages, effort=effort)
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                stop=renderer.get_stop_sequences(),
            ),
        )
        message, termination = renderer.parse_response(response.sequences[0].tokens)
        text = get_text_content(message)
        try:
            return AuditResult.from_model_text(
                text,
                expected_instance_id=instance.instance_id,
                termination=termination,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            messages.extend(
                [
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=f"The response could not be parsed: {error}. Return corrected JSON only.",
                    ),
                ]
            )
    raise RuntimeError(f"failed to audit {instance.instance_id}: {last_error}")


async def _refine_one(
    instance: KokkosInstance,
    first_pass: dict[str, object],
    *,
    sampling_client: tinker.SamplingClient,
    renderer: TmlV0Renderer,
    effort: float,
    max_tokens: int,
    attempts: int,
) -> AuditResult:
    if first_pass["assessment"] == "clear":
        return AuditResult(
            instance_id=instance.instance_id,
            assessment="clear",
            issues=tuple(str(item) for item in first_pass["issues"]),
            revised_problem_statement=instance.problem_statement.strip(),
            confidence=float(first_pass["confidence"]),
            termination=str(first_pass["termination"]),
        )
    messages = [
        Message(role="system", content=REFINEMENT_SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"INSTANCE ID: {instance.instance_id}\n\n"
                "ORIGINAL STATEMENT:\n"
                f"{instance.problem_statement.strip()}\n\n"
                "FIRST-PASS ISSUES:\n"
                f"{json.dumps(first_pass['issues'])}\n\n"
                "OVERFIT FIRST-PASS DRAFT:\n"
                f"{first_pass['revised_problem_statement']}"
            ),
        ),
    ]
    last_error: Exception | None = None
    for _attempt in range(attempts):
        prompt = renderer.build_generation_prompt(messages, effort=effort)
        response = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                stop=renderer.get_stop_sequences(),
            ),
        )
        message, termination = renderer.parse_response(response.sequences[0].tokens)
        text = get_text_content(message)
        try:
            result = AuditResult.from_model_text(
                text,
                expected_instance_id=instance.instance_id,
                termination=termination,
            )
            if len(result.revised_problem_statement.split()) > 300:
                raise ValueError("revised_problem_statement exceeds 300 words")
            return result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            messages.extend(
                [
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=f"The response failed validation: {error}. Return corrected JSON only.",
                    ),
                ]
            )
    raise RuntimeError(f"failed to refine {instance.instance_id}: {last_error}")


def _apply_reports(
    instances_path: Path,
    reports_dir: Path,
    output_dir: Path,
    *,
    org: str,
) -> tuple[int, int]:
    instances = _read_instances(instances_path)
    updated: list[KokkosInstance] = []
    revised = 0
    invalid = 0
    for instance in instances:
        report_path = reports_dir / f"{instance.instance_id}.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing audit report: {report_path}")
        report = json.loads(report_path.read_text())
        assessment = str(report["assessment"])
        if assessment == "invalid":
            invalid += 1
        statement = instance.problem_statement
        if assessment == "needs_revision":
            statement = str(report["revised_problem_statement"]).strip()
            revised += statement != instance.problem_statement.strip()
        value = instance.to_dict()
        value["problem_statement"] = statement
        updated.append(KokkosInstance.from_dict(value))

    content = "".join(json.dumps(instance.to_dict(), sort_keys=True) + "\n" for instance in updated)
    instances_path.write_text(content)
    for instance in updated:
        export_instance(instance, output_dir, org=org)
    return revised, invalid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("data/kokkos/SWE-kokkos-bench-v2/instances.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kokkos/SWE-kokkos-bench-v2"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("notes/experiments/SWE-kokkos-bench/instruction-audit/inkling"),
    )
    parser.add_argument("--model-name", default="thinkingmachines/Inkling:peft:262144")
    parser.add_argument("--thinking-effort", type=float, default=0.99)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-concurrency", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--org", default="luosuu")
    parser.add_argument(
        "--refine-from",
        type=Path,
        help="directory of first-pass reports to edit without resending private patches",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    if not math.isfinite(args.thinking_effort) or not 0.0 <= args.thinking_effort < 1.0:
        raise ValueError("thinking_effort must be finite and in [0, 1)")
    if args.apply:
        revised, invalid = _apply_reports(
            args.instances, args.reports_dir, args.output_dir, org=args.org
        )
        print(f"Applied {revised} revised statements; {invalid} tasks remain marked invalid")
        return

    _load_env_file(args.env_file)
    instances = _read_instances(args.instances)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    service_client = tinker.ServiceClient(
        user_metadata=recipe_user_metadata("kokkos_instruction_audit")
    )
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.model_name
    )
    renderer_name = model_info.get_recommended_renderer_name(args.model_name)
    renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(args.model_name))
    if not isinstance(renderer, TmlV0Renderer):
        raise TypeError(f"instruction audit requires TmlV0Renderer, got {type(renderer).__name__}")
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def run_one(instance: KokkosInstance) -> AuditResult:
        report_path = args.reports_dir / f"{instance.instance_id}.json"
        if report_path.is_file():
            print(f"{instance.instance_id}: SKIP")
            value = json.loads(report_path.read_text())
            return AuditResult(
                instance_id=str(value["instance_id"]),
                assessment=str(value["assessment"]),
                issues=tuple(str(item) for item in value["issues"]),
                revised_problem_statement=str(value["revised_problem_statement"]),
                confidence=float(value["confidence"]),
                termination=str(value["termination"]),
            )
        async with semaphore:
            if args.refine_from is None:
                result = await _audit_one(
                    instance,
                    sampling_client=sampling_client,
                    renderer=renderer,
                    effort=args.thinking_effort,
                    max_tokens=args.max_tokens,
                    attempts=args.attempts,
                )
            else:
                first_pass_path = args.refine_from / f"{instance.instance_id}.json"
                if not first_pass_path.is_file():
                    raise FileNotFoundError(f"missing first-pass report: {first_pass_path}")
                result = await _refine_one(
                    instance,
                    json.loads(first_pass_path.read_text()),
                    sampling_client=sampling_client,
                    renderer=renderer,
                    effort=args.thinking_effort,
                    max_tokens=args.max_tokens,
                    attempts=args.attempts,
                )
        report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"{instance.instance_id}: {result.assessment}")
        return result

    results = await asyncio.gather(*(run_one(instance) for instance in instances))
    counts = dict.fromkeys(("clear", "needs_revision", "invalid"), 0)
    for result in results:
        counts[result.assessment] += 1
    summary = {
        "model_name": args.model_name,
        "thinking_effort": args.thinking_effort,
        "temperature": 1.0,
        "max_tokens": args.max_tokens,
        "instance_count": len(results),
        "counts": counts,
    }
    (args.reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
