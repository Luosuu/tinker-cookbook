"""Versioned, task-independent workflow shared by sampling, RL and Pi."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask

HPC_DEBUG_V1 = """HPC debugging workflow (hpc-debug-v1)
Work in the supplied repository and use only public source and local build evidence.
Do not access hidden verifier files, reference solutions, or upstream PR history.

1. Orient once: identify the repository, compiler, configured execution backend,
   CMake cache, existing build directory, and relevant test targets. Reuse the
   existing configuration. Respect the environment's CPU and memory allocation;
   do not choose build parallelism from host CPU count alone.
2. State one concrete bug hypothesis and the smallest observation that can
   falsify it. Search the relevant source subtree with line numbers. For template
   traits, also inspect macro expansions, aliases, includes, and generated headers;
   failure to find a literal definition is not evidence that it is missing.
3. Reproduce with the smallest relevant compile/test target. Preserve compiler
   flags, include paths, and backend. Capture complete logs in a temporary file
   and inspect the first causal diagnostic plus the final status. Do not rerun
   the same failing command unless code, configuration, or the hypothesis changed.
4. Make a minimal source fix. Check execution/memory-space accessibility, fences
   and asynchronous lifetimes, ownership, layout/strides, team/vector semantics,
   and host/device annotations when relevant. A Serial pass does not establish
   CUDA/HIP/OpenMP correctness; report which backends were actually exercised.
5. Run the focused reproducer, then related regression tests if budget permits.
   Distinguish compile failure, assertion failure, timeout/OOM, and unavailable
   hardware. Never weaken tests or treat an unavailable backend as a pass.
6. After repeated unproductive searches, summarize evidence and change strategy.
   Keep a short local note of hypothesis, changed files, commands and outcomes.
   Reserve time for validation and git diff --check. Once the fix is supported,
   finish with a concise summary of the change, tests and remaining limitations;
   do not spend the remaining budget on unrelated exploration.
"""


def profile_text(name: str) -> str:
    if name == "baseline":
        return ""
    if name == "hpc-debug-v1":
        return HPC_DEBUG_V1
    raise ValueError(f"Unknown harness profile: {name!r}")


def apply_profile(task: HarborTask, name: str) -> HarborTask:
    """Keep task payloads intact; bind exact workflow bytes into eval identity."""
    text = profile_text(name)
    if not text:
        return task
    return replace(
        task,
        instruction=f"{text}\nTask\n====\n{task.instruction}",
        config={
            **task.config,
            "harness": {"profile": name, "sha256": hashlib.sha256(text.encode()).hexdigest()},
        },
    )
