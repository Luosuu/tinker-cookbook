"""OpenHands execution adapter for prebuilt Apptainer agent-server SIF images.

Requires openhands-workspace 1.45.0 and Apptainer on the machine running the
factory. Image construction and Slurm resource allocation happen separately.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import secrets
import shlex
import socket
import threading
import time
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

if TYPE_CHECKING:
    from openhands.workspace import ApptainerWorkspace


_PORT_LOCK = threading.Lock()
_LEASED_PORTS: set[int] = set()


def _lease_port() -> int:
    with _PORT_LOCK:
        for _ in range(1000):
            # Check the lease before bind: even a temporary probe can race with
            # a different constructor trying to bind its already-leased port.
            port = 10000 + secrets.randbelow(20000)
            if port in _LEASED_PORTS:
                continue
            with socket.socket() as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
            _LEASED_PORTS.add(port)
            return port
    raise RuntimeError("Could not reserve a unique sandbox port")


def _release_port(port: int) -> None:
    with _PORT_LOCK:
        _LEASED_PORTS.discard(port)


class ApptainerSandbox:
    """One persistent container per episode, with async Cookbook operations."""

    def __init__(
        self,
        workspace: ApptainerWorkspace,
        timeout: int,
        default_workdir: str | None = None,
        *,
        allow_network: bool = True,
        leased_port: int | None = None,
    ) -> None:
        self._workspace = workspace
        self._id = f"apptainer-{uuid.uuid4().hex}"
        self._deadline = time.monotonic() + timeout
        self._closed = False
        self._lock = asyncio.Lock()
        self._default_workdir = default_workdir
        self._allow_network = allow_network
        self._leased_port = leased_port

    @classmethod
    async def create(
        cls,
        sif_file: str | Path,
        timeout: int = 1800,
        default_workdir: str | None = None,
        *,
        enable_gpu: bool = False,
        allow_network: bool = True,
    ) -> ApptainerSandbox:
        from openhands.workspace import ApptainerWorkspace

        if (
            enable_gpu
            and "SLURM_JOB_ID" in os.environ
            and not os.environ.get("CUDA_VISIBLE_DEVICES")
        ):
            raise RuntimeError(
                "GPU sandbox requires Slurm CUDA_VISIBLE_DEVICES; request GPUs first"
            )
        forward_env = ["SESSION_API_KEY", "OH_ENABLE_VSCODE"]
        if enable_gpu and "CUDA_VISIBLE_DEVICES" in os.environ:
            forward_env.append("CUDA_VISIBLE_DEVICES")

        # Set once for this worker process; never forward the Tinker API key.
        os.environ.setdefault("SESSION_API_KEY", secrets.token_urlsafe(32))
        os.environ["OH_ENABLE_VSCODE"] = "false"
        sif_path = str(Path(sif_file).resolve(strict=True))

        def start_once() -> ApptainerSandbox:
            port = _lease_port()
            marker_name = "OH_SANDBOX_" + uuid.uuid4().hex.upper()
            marker_value = secrets.token_hex(32)
            workspace = None
            os.environ[marker_name] = marker_value
            try:
                workspace = ApptainerWorkspace(
                    sif_file=sif_path,
                    host_port=port,
                    cache_dir=os.environ.get("APPTAINER_CACHEDIR"),
                    use_fakeroot=True,
                    enable_docker_compat=True,
                    enable_gpu=enable_gpu,
                    disable_mount_locations=["hostfs", "bind-paths", "cwd", "home"],
                    forward_env=[*forward_env, marker_name],
                    health_check_timeout=180,
                )
                # Generic /health can succeed against an unrelated listener.
                identity = workspace.execute_command(f"printenv {marker_name}", cwd="/", timeout=30)
                if identity.exit_code != 0 or identity.stdout.strip() != marker_value:
                    raise RuntimeError("Sandbox endpoint identity mismatch")
                if not allow_network:
                    probe = workspace.execute_command(
                        "unshare --user --map-root-user --net -- /bin/true", cwd="/", timeout=30
                    )
                    if probe.exit_code != 0:
                        raise RuntimeError("Offline command namespace unavailable: " + probe.stderr)
                return cls(
                    workspace,
                    timeout,
                    default_workdir,
                    allow_network=allow_network,
                    leased_port=port,
                )
            except BaseException:
                if workspace is not None:
                    # This kills this workspace's local process, not the remote endpoint.
                    workspace.cleanup()
                _release_port(port)
                raise
            finally:
                os.environ.pop(marker_name, None)

        def start() -> ApptainerSandbox:
            for attempt in range(8):
                try:
                    return start_once()
                except RuntimeError as error:
                    # The upstream pre-bind check happens before spawning a server.
                    # A port can become busy after our probe; reserve a new one.
                    if (
                        not re.fullmatch(r"Port [0-9]+ is not available", str(error))
                        or attempt == 7
                    ):
                        raise
                    time.sleep(0.1)
            raise AssertionError("unreachable")

        startup = asyncio.create_task(asyncio.to_thread(start))
        try:
            return await asyncio.shield(startup)
        except asyncio.CancelledError:
            # Cancelling an asyncio future does not stop its constructor thread.
            try:
                sandbox = await startup
                await sandbox.cleanup()
            except Exception:
                pass
            raise

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
            if not self._allow_network:
                command = "unshare --user --map-root-user --net -- /bin/bash -c " + shlex.quote(
                    command
                )
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
            if self._leased_port is not None:
                _release_port(self._leased_port)


async def apptainer_sandbox_factory(
    env_dir: Path, timeout: int, *, allow_network: bool = True
) -> ApptainerSandbox:
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
    gpus = config.get("environment", {}).get("gpus", 0)
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise ValueError("environment.gpus must be a nonnegative integer")
    # Slurm allocates devices separately. This flag grants visibility, not a quota.
    return await ApptainerSandbox.create(
        sif, timeout, default_workdir=workdir, enable_gpu=gpus > 0, allow_network=allow_network
    )
