"""OpenHands execution adapter for prebuilt Apptainer agent-server SIF images.

Requires openhands-workspace 1.45.0 and Apptainer on the machine running the
factory. Image construction and Slurm resource allocation happen separately.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import shlex
import socket
import time
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

if TYPE_CHECKING:
    from openhands.workspace import ApptainerWorkspace


class ApptainerSandbox:
    """One persistent container per episode, with async Cookbook operations."""

    def __init__(
        self, workspace: ApptainerWorkspace, timeout: int, default_workdir: str | None = None
    ) -> None:
        self._workspace = workspace
        self._id = f"apptainer-{uuid.uuid4().hex}"
        self._deadline = time.monotonic() + timeout
        self._closed = False
        self._lock = asyncio.Lock()
        self._default_workdir = default_workdir

    @classmethod
    async def create(
        cls, sif_file: str | Path, timeout: int = 1800, default_workdir: str | None = None
    ) -> ApptainerSandbox:
        from openhands.workspace import ApptainerWorkspace

        # Set once for this worker process; never forward the Tinker API key.
        os.environ.setdefault("SESSION_API_KEY", secrets.token_urlsafe(32))
        os.environ["OH_ENABLE_VSCODE"] = "false"
        workspace = await asyncio.to_thread(
            ApptainerWorkspace,
            sif_file=str(Path(sif_file).resolve(strict=True)),
            cache_dir=os.environ.get("APPTAINER_CACHEDIR"),
            use_fakeroot=True,
            enable_docker_compat=True,
            forward_env=["SESSION_API_KEY", "OH_ENABLE_VSCODE"],
            health_check_timeout=180,
        )
        return cls(workspace, timeout, default_workdir)

    @property
    def sandbox_id(self) -> str:
        return self._id

    async def send_heartbeat(self, timeout: int = 30) -> None:
        if self._closed:
            raise SandboxTerminatedError(self._id)

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        limit = 128 * 1024 if max_output_bytes is None else max_output_bytes
        if limit < 0:
            raise ValueError("max_output_bytes must be nonnegative")
        async with self._lock:
            remaining = self._deadline - time.monotonic()
            if self._closed or remaining <= 0:
                raise SandboxTerminatedError(self._id)
            result = await asyncio.to_thread(
                self._workspace.execute_command,
                command,
                cwd=workdir if workdir is not None else self._default_workdir,
                timeout=min(timeout, remaining),
            )
        return SandboxResult(
            stdout=result.stdout.encode()[:limit].decode(errors="replace"),
            stderr=result.stderr.encode()[:limit].decode(errors="replace"),
            exit_code=result.exit_code,
        )

    async def read_file(
        self,
        path: str,
        max_bytes: int | None = None,
        timeout: int = 60,
    ) -> SandboxResult:
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        command = (
            f"cat -- {shlex.quote(path)}"
            if max_bytes is None
            else (f"head -c {max_bytes} -- {shlex.quote(path)}")
        )
        return await self.run_command(command, timeout=timeout, max_output_bytes=max_bytes)

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        encoded = base64.b64encode(data).decode("ascii")
        program = (
            "import base64,pathlib; "
            f"p=pathlib.Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            f"p.write_bytes(base64.b64decode({encoded!r})); "
            + ("p.chmod(p.stat().st_mode | 0o111)" if executable else "pass")
        )
        return await self.run_command("python -c " + shlex.quote(program), timeout=timeout)

    async def cleanup(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await asyncio.to_thread(self._workspace.cleanup)
            deadline = time.monotonic() + 20
            while True:
                with socket.socket() as sock:
                    sock.settimeout(0.2)
                    listening = sock.connect_ex(("127.0.0.1", self._workspace.host_port)) == 0
                if not listening:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Sandbox listener survived cleanup: {self._id}")
                await asyncio.sleep(0.2)


async def apptainer_sandbox_factory(env_dir: Path, timeout: int) -> ApptainerSandbox:
    """Read an explicit SIF path; do not silently replace task Dockerfiles.

    The environment must contain agent-server.sif or a sif.path file naming a
    prebuilt task image containing the OpenHands agent server and its entrypoint.
    """
    pointer = env_dir / "sif.path"
    sif = Path(pointer.read_text().strip()) if pointer.exists() else env_dir / "agent-server.sif"
    if not sif.is_absolute():
        sif = (env_dir / sif).resolve() if pointer.exists() else sif.resolve()
    config_path = env_dir.parent / "task.toml"
    config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    workdir = config.get("environment", {}).get("workdir")
    if workdir is not None and (not isinstance(workdir, str) or not workdir.startswith("/")):
        raise ValueError("environment.workdir must be an absolute container path")
    return await ApptainerSandbox.create(sif, timeout, default_workdir=workdir)
