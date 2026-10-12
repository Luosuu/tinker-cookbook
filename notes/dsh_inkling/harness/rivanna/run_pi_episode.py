"""Run one Pi HPC agent episode on a Rivanna Apptainer sandbox.

Uses the cookbook's `eval_kokkos_pi.pi_rollout` with the sandbox opened through
`apptainer.open_sandbox`, so images, preflight, threads and grading match the native and DSH
arms. Infrastructure failures are retried from a fresh sandbox exactly as for the native
arm. Writes `summary.json` in the layout `compare_runs` reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# Keys a block arm may set; everything else is the matched native budget.
PI_OPTIONS = {
    "hpc_mode",
    "hpc_skill",
    "plugin_profile",
    "max_protocol_repairs",
    "guidance",
    "file_tools",
    "build_check",
    "symbol_brief",
}


def pi_config(*, pi_agent_dir: Path, budget: dict, options: dict):
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_pi import CLIConfig

    unknown = set(options) - PI_OPTIONS
    if unknown:
        raise ValueError(f"unsupported Pi arm options: {sorted(unknown)}")
    if "max_protocol_repairs" in options:
        options = {**options, "max_protocol_repairs": int(options["max_protocol_repairs"])}
    return CLIConfig(
        pi_agent_dir=str(pi_agent_dir),
        model_name="thinkingmachines/Inkling-Small:peft:262144",
        thinking_effort=0.9,
        temperature=1.0,
        context_window=262144,
        max_wall_seconds=budget["wall_seconds"],
        max_turns=budget["max_model_requests"],
        max_tool_calls=budget["max_tool_calls"],
        max_tokens=16384,
        max_sampled_tokens=budget["max_sampled_tokens"],
        max_input_tokens=budget["max_input_tokens"],
        max_context_tokens=budget["max_context_tokens"],
        max_tool_output_chars=16384,
        sandbox_timeout=budget["sandbox_timeout"],
        command_timeout=budget["command_timeout"],
        grader_timeout=budget["grader_timeout"],
        build_check_max_runs=budget["build_check_max_runs"],
        build_check_max_seconds=budget["build_check_max_seconds"],
        **options,
    )


async def run(
    *,
    task: str,
    output: Path,
    tasks_root: Path,
    code_root: Path,
    env_file: Path,
    pi_agent_dir: Path,
    budget: dict,
    options: dict,
) -> dict:
    sys.path.insert(0, str(code_root.resolve()))
    from notes.dsh_inkling.harness.rivanna.apptainer import load_guarded_task, open_sandbox
    from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
    from tinker_cookbook.recipes.kokkos_rl.rl.candidate_artifact import capture_candidate
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos import load_env_file
    from tinker_cookbook.recipes.kokkos_rl.rl.eval_kokkos_pi import harness_identity, pi_rollout
    from tinker_cookbook.utils.ml_log import dump_config

    load_env_file(env_file)
    harbor_task = load_guarded_task(tasks_root, task)
    config = pi_config(pi_agent_dir=pi_agent_dir, budget=budget, options=options)
    grader_timeout = budget["grader_timeout"]
    sandbox_timeout = budget["sandbox_timeout"]
    max_infra_retries = budget["native_max_infra_retries"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "harness": "pi",
                "sandbox": "apptainer",
                "sif_path": (harbor_task.task_dir / "environment/sif.path").read_text().strip(),
                "task": task,
                "max_infra_retries": max_infra_retries,
                "pi_config": dump_config(config),
                "pi_identity": harness_identity(config),
            },
            indent=2,
        )
    )
    started = time.monotonic()
    attempts: list[dict] = []
    for attempt in range(max_infra_retries + 1):
        directory = output / f"attempt-{attempt}"
        directory.mkdir()
        record: dict = {"attempt": attempt, "error": None}
        sandbox = None
        try:
            sandbox = await open_sandbox(
                harbor_task, sandbox_timeout, hidden_paths=(str(env_file.resolve()),)
            )
            rollout = await pi_rollout(config, harbor_task, sandbox, directory)
            await capture_candidate(sandbox, task=harbor_task, results_dir=directory)
            reward, details = await HarborReward(
                tests_dir=harbor_task.task_dir / "tests",
                sandbox=sandbox,
                grader_timeout=grader_timeout,
                raise_on_grading_error=True,
                task_name=task,
                grading_log_path=directory / "grading.json",
            )([])
            record.update(
                reward=reward,
                reward_details=details,
                turns_used=rollout["turns"],
                tool_calls=rollout["tool_calls"],
                stop_reason=rollout["stop_reason"],
                completion_kind=rollout.get("completion_kind"),
                input_tokens=rollout["input_tokens"],
                output_tokens=rollout["output_tokens"],
            )
        except Exception as exc:
            record.update(reward=0.0, error=f"{type(exc).__name__}: {exc}")
        finally:
            if sandbox is not None:
                await sandbox.cleanup()
        attempts.append(record)
        if record["error"] is None:
            break
    final = output / f"attempt-{len(attempts) - 1}"
    for name in final.iterdir():
        target = output / name.name
        if not target.exists():
            (shutil.copytree if name.is_dir() else shutil.copy2)(name, target)
    with (output / "attempts.jsonl").open("w") as stream:
        for record in attempts:
            stream.write(json.dumps(record) + "\n")
    summary = {
        "task_name": task,
        **{k: v for k, v in attempts[-1].items() if k != "attempt"},
        "harness": "pi",
        "harness_status": attempts[-1].get("stop_reason") or "error",
        "attempts": len(attempts),
        "time_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--pi-agent-dir", type=Path, required=True)
    parser.add_argument(
        "--budget", action="append", default=[], help="key=value override of EPISODE_BUDGET"
    )
    parser.add_argument("--option", action="append", default=[], help="key=value Pi arm option")
    args = parser.parse_args()
    options = dict(item.split("=", 1) for item in args.option)
    del args.option
    from notes.dsh_inkling.harness.rivanna.budget import resolve_budget

    args.budget = resolve_budget(args.budget)
    print(json.dumps(asyncio.run(run(**vars(args), options=options)), indent=2, default=str))


if __name__ == "__main__":
    main()
