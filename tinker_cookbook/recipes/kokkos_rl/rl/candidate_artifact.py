"""Retain a candidate patch before hidden tests alter the sandbox worktree."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import time
from pathlib import Path

from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
from tinker_cookbook.sandbox import SandboxInterface

EXPORT_SCRIPT = r"""
import base64, hashlib, json, subprocess, sys
baseline = sys.argv[1]
def git(*args, expected=(0,)):
    p = subprocess.run(["git", "--no-replace-objects", *args], capture_output=True)
    if p.returncode not in expected:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout
git("cat-file", "-e", baseline + "^{commit}")
patch = git("diff", "--no-ext-diff", "--no-textconv", "--binary", baseline, "--")
for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
    if name and not name.startswith(b"build/"):
        patch += git("diff", "--no-ext-diff", "--no-textconv", "--binary", "--no-index", "--", "/dev/null", name.decode(), expected=(0, 1))
if len(patch) > 4 * 1024 * 1024:
    raise RuntimeError("Candidate exceeds the 4 MiB artifact limit")
print(json.dumps({"base_commit": baseline, "head_commit": git("rev-parse", "HEAD").decode().strip(), "patch_bytes": len(patch), "patch_sha256": hashlib.sha256(patch).hexdigest(), "patch_base64": base64.b64encode(patch).decode()}))
"""


def export_command(baseline: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", baseline):
        raise ValueError("Candidate export requires an explicit baseline commit")
    return f"python3 -c {shlex.quote(EXPORT_SCRIPT)} {baseline}"


async def capture_candidate(
    sandbox: SandboxInterface, *, task: HarborTask, results_dir: Path
) -> None:
    start = time.monotonic()
    metadata: dict[str, object] = {
        "stage": "before_hidden_tests",
        "sandbox_id": sandbox.sandbox_id,
        "complete": False,
    }
    try:
        matches = re.findall(
            r"^baseline=([0-9a-f]{40})$",
            (task.task_dir / "tests/test.sh").read_text(),
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError("Task verifier does not identify one immutable baseline")
        result = await sandbox.run_command(
            export_command(matches[0]),
            workdir="/workspace/repo",
            timeout=60,
            max_output_bytes=6 * 1024 * 1024,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Candidate export failed: {result.stderr}")
        record = json.loads(result.stdout)
        patch = base64.b64decode(record.pop("patch_base64"), validate=True)
        if (
            len(patch) != record["patch_bytes"]
            or hashlib.sha256(patch).hexdigest() != record["patch_sha256"]
        ):
            raise ValueError("Candidate artifact was truncated or changed in transit")
        (results_dir / "candidate.patch").write_bytes(patch)
        metadata.update(record)
        metadata["complete"] = True
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    metadata["capture_seconds"] = round(time.monotonic() - start, 3)
    (results_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
