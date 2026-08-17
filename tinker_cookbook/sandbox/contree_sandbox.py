"""Nebius ConTree-backed sandbox implementation.

ConTree commands create immutable filesystem versions. A ConTree session keeps
the latest version, giving callers the persistent-container semantics expected
by :class:`SandboxInterface` without keeping a VM alive between commands.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import uuid
from datetime import timedelta
from pathlib import Path

from contree_sdk import Contree
from contree_sdk.auth import IAMAuth, JWTAuth
from contree_sdk.config import ContreeConfig
from contree_sdk.sdk.objects.image._async import ContreeImage
from contree_sdk.sdk.objects.session._async import ContreeSession
from contree_sdk.utils.models.file import UploadFileSpec

from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError

DEFAULT_CONTREE_IMAGE = "python:3.12-slim"
DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024


def create_contree_client(timeout: int) -> Contree:
    """Create a client, including the ``.env`` key name used by this project."""

    base_url = os.environ.get("CONTREE_BASE_URL")
    legacy_token = os.environ.get("CONTREE_TOKEN")
    if legacy_token:
        auth = JWTAuth(token=legacy_token)
        if base_url:
            auth.base_url = base_url
    else:
        token = os.environ.get("NEBIUS_SANDBOX_API_KEY", "NEBIUS_API_KEY")
        auth = IAMAuth(token=token)
        if base_url:
            auth.base_url = base_url
    return Contree(
        ContreeConfig(
            auth=auth,
            operation_run_timeout=float(timeout),
            operation_timeout=float(timeout),
            default_truncate_output_at=DEFAULT_MAX_OUTPUT_BYTES,
        )
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


class ContreeSandbox:
    """Persistent sandbox implemented with a ConTree image session."""

    def __init__(
        self,
        *,
        client: Contree,
        image: ContreeImage,
        timeout: int,
        max_stream_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        default_workdir: str | None = None,
    ) -> None:
        self._client = client
        self._session: ContreeSession = image.session()
        self._timeout = timeout
        self._max_stream_output_bytes = max_stream_output_bytes
        self._default_workdir = default_workdir
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        image: str = DEFAULT_CONTREE_IMAGE,
        timeout: int = 600,
        max_stream_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        client: Contree | None = None,
        import_image: bool = True,
        default_workdir: str | None = None,
    ) -> ContreeSandbox:
        """Create a sandbox from an OCI reference or existing ConTree UUID."""

        client = client or create_contree_client(timeout)
        token_info = await client.get_token_info()
        required_permissions = {"spawn"}
        if import_image:
            required_permissions.add("import")
        missing_permissions = sorted(
            permission
            for permission in required_permissions
            if not token_info.permissions.get(permission, False)
        )
        if missing_permissions:
            raise PermissionError(
                "ConTree credential lacks required permissions "
                f"{missing_permissions}; set NEBIUS_PROJECT_ID to the project authorized for "
                "this API key and rotate the key if its permissions are all false"
            )
        contree_image = (
            await client.images.oci(image, timeout=timeout)
            if import_image
            else await client.images.use(image)
        )
        return cls(
            client=client,
            image=contree_image,
            timeout=timeout,
            max_stream_output_bytes=max_stream_output_bytes,
            default_workdir=default_workdir,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise SandboxTerminatedError("ConTree sandbox has been cleaned up")

    @property
    def sandbox_id(self) -> str:
        return str(self._session.uuid or self._session.tag or "unresolved")

    async def send_heartbeat(self, timeout: int = 30) -> None:
        del timeout
        self._ensure_open()
        # ConTree persists image state and has no live-container lease to refresh.

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        self._ensure_open()
        cap = max_output_bytes if max_output_bytes is not None else self._max_stream_output_bytes
        try:
            result = await self._session.run(
                shell=command,
                cwd=workdir or self._default_workdir,
                timeout=timedelta(seconds=min(timeout, self._timeout)),
                disposable=False,
                truncate_output_at=cap,
            )
            return SandboxResult(
                stdout=_as_text(result.stdout),
                stderr=_as_text(result.stderr),
                exit_code=result.exit_code,
                metrics={"elapsed_seconds": result.elapsed.total_seconds()},
            )
        except Exception as error:
            return SandboxResult(stdout="", stderr=str(error), exit_code=-1)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        self._ensure_open()
        try:
            content = await asyncio.wait_for(self._session.read(path), timeout=timeout)
            if max_bytes is not None:
                content = content[:max_bytes]
            return SandboxResult(
                stdout=content.decode("utf-8", errors="replace"),
                stderr="",
                exit_code=0,
            )
        except Exception as error:
            return SandboxResult(stdout="", stderr=str(error), exit_code=-1)

    async def write_file(
        self,
        path: str,
        content: str | bytes = "",
        executable: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        self._ensure_open()
        payload = content.encode() if isinstance(content, str) else content
        mode = 0o755 if executable else 0o644
        try:
            result = await self._session.run(
                shell="true",
                files=[UploadFileSpec(source=payload, path=path, mode=mode)],
                timeout=timedelta(seconds=min(timeout, self._timeout)),
                disposable=False,
                truncate_output_at=self._max_stream_output_bytes,
            )
            return SandboxResult(
                stdout=_as_text(result.stdout),
                stderr=_as_text(result.stderr),
                exit_code=result.exit_code,
                metrics={"elapsed_seconds": result.elapsed.total_seconds()},
            )
        except Exception as error:
            return SandboxResult(stdout="", stderr=str(error), exit_code=-1)

    async def cleanup(self) -> None:
        # Completed ConTree operations are immutable image versions; there is no
        # live resource to terminate.
        self._closed = True


def _dockerfile_instructions(path: Path) -> list[tuple[str, str]]:
    """Parse the small Dockerfile subset used by exported Harbor tasks."""

    logical_lines: list[str] = []
    current = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current += (" " if current else "") + line.removesuffix("\\").rstrip()
        if not line.endswith("\\"):
            logical_lines.append(current)
            current = ""
    if current:
        logical_lines.append(current)

    instructions: list[tuple[str, str]] = []
    for line in logical_lines:
        instruction, _, value = line.partition(" ")
        instruction = instruction.upper()
        if instruction not in {"FROM", "ENV", "RUN", "WORKDIR"}:
            raise ValueError(f"unsupported Dockerfile instruction {instruction!r} in {path}")
        instructions.append((instruction, value.strip()))
    return instructions


class ContreeDockerfileSandboxFactory:
    """Prepare each Harbor Dockerfile once, then branch isolated ConTree sessions.

    ConTree imports OCI base images but does not build local Dockerfiles. Harbor's
    exported task images use a deliberately small Dockerfile subset, so this
    factory executes their RUN instructions once and persists the resulting image
    UUIDs for resumable evaluations.
    """

    def __init__(self, cache_path: Path, timeout: int = 3600) -> None:
        self._cache_path = cache_path
        self._timeout = timeout
        self._client = create_contree_client(timeout)
        self._cache_lock = asyncio.Lock()
        self._prepare_locks: dict[str, asyncio.Lock] = {}
        self._prepared_images = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not self._cache_path.is_file():
            return {}
        data = json.loads(self._cache_path.read_text())
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            raise ValueError(f"invalid ConTree image cache: {self._cache_path}")
        return data

    async def _store_cache(self) -> None:
        async with self._cache_lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(self._prepared_images, indent=2, sort_keys=True))
            temporary.replace(self._cache_path)

    async def _prepare(self, dockerfile_path: Path, timeout: int) -> tuple[str, str | None]:
        contents = dockerfile_path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        instructions = _dockerfile_instructions(dockerfile_path)
        workdir = next(
            (value for instruction, value in reversed(instructions) if instruction == "WORKDIR"),
            None,
        )
        if digest in self._prepared_images:
            return self._prepared_images[digest], workdir

        lock = self._prepare_locks.setdefault(digest, asyncio.Lock())
        async with lock:
            if digest in self._prepared_images:
                return self._prepared_images[digest], workdir

            base_images = [value for instruction, value in instructions if instruction == "FROM"]
            if len(base_images) != 1:
                raise ValueError(
                    f"expected one FROM instruction in {dockerfile_path}, got {len(base_images)}"
                )
            environment: list[str] = []
            sandbox = await ContreeSandbox.create(
                image=base_images[0],
                timeout=min(timeout, self._timeout),
                client=self._client,
            )
            try:
                for instruction, value in instructions:
                    if instruction == "ENV":
                        environment.append(value)
                    elif instruction == "RUN":
                        exports = " ".join(shlex.quote(item) for item in environment)
                        prefix = f"export {exports}; " if exports else ""
                        result = await sandbox.run_command(
                            "set -eu; " + prefix + value,
                            timeout=min(timeout, self._timeout),
                        )
                        if result.exit_code != 0:
                            raise RuntimeError(
                                f"failed to prepare {dockerfile_path.parent.parent.name}: "
                                f"exit={result.exit_code}\n{result.stdout[-16000:]}\n"
                                f"{result.stderr[-16000:]}"
                            )
                image_id = sandbox.sandbox_id
                self._prepared_images[digest] = image_id
                await self._store_cache()
                return image_id, workdir
            finally:
                await sandbox.cleanup()

    async def __call__(self, env_dir: Path, timeout: int) -> ContreeSandbox:
        image_id, workdir = await self._prepare(env_dir / "Dockerfile", timeout)
        return await ContreeSandbox.create(
            image=image_id,
            timeout=min(timeout, self._timeout),
            client=self._client,
            import_image=False,
            default_workdir=workdir,
        )


class ContreeSandboxPool:
    """Branch concurrent one-shot sandboxes from one prepared ConTree image."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_CONTREE_IMAGE,
        setup_command: str | None = None,
        pool_size: int | None = None,
        sandbox_timeout_secs: int = 1200,
    ) -> None:
        self._image = image
        self._setup_command = setup_command
        self._pool_size = pool_size or int(os.environ.get("CONTREE_POOL_SIZE", "32"))
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._client = create_contree_client(sandbox_timeout_secs)
        self._prepared_image_id: str | None = None
        self._prepare_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._pool_size)
        self._terminated = False

    async def _prepare(self) -> str:
        if self._prepared_image_id is not None:
            return self._prepared_image_id
        async with self._prepare_lock:
            if self._prepared_image_id is not None:
                return self._prepared_image_id
            sandbox = await ContreeSandbox.create(
                image=self._image,
                timeout=self._sandbox_timeout_secs,
                client=self._client,
            )
            if self._setup_command:
                result = await sandbox.run_command(
                    self._setup_command,
                    timeout=self._sandbox_timeout_secs,
                )
                if result.exit_code != 0:
                    raise RuntimeError(
                        "Failed to prepare ConTree image: "
                        f"exit={result.exit_code}\n{result.stdout}\n{result.stderr}"
                    )
            self._prepared_image_id = sandbox.sandbox_id
            await sandbox.cleanup()
            return self._prepared_image_id

    async def run_in_workdir(
        self,
        files: dict[str, str],
        command: list[str],
        timeout: int | None = None,
    ) -> SandboxResult:
        if self._terminated:
            raise SandboxTerminatedError("ContreeSandboxPool has been terminated")
        async with self._semaphore:
            image_id = await self._prepare()
            sandbox = await ContreeSandbox.create(
                image=image_id,
                timeout=self._sandbox_timeout_secs,
                client=self._client,
                import_image=False,
            )
            workdir = f"/workspace/{uuid.uuid4().hex[:12]}"
            try:
                mkdir = await sandbox.run_command(
                    f"mkdir -p {shlex.quote(workdir)}", timeout=timeout or 60
                )
                if mkdir.exit_code != 0:
                    return mkdir
                # A ConTree session advances one immutable image version at a
                # time, so mutations on the same session must be sequential.
                for name, content in files.items():
                    write = await sandbox.write_file(f"{workdir}/{name}", content)
                    if write.exit_code != 0:
                        return write
                return await sandbox.run_command(
                    shlex.join(command),
                    workdir=workdir,
                    timeout=timeout or self._sandbox_timeout_secs,
                )
            finally:
                await sandbox.cleanup()

    async def terminate(self) -> None:
        self._terminated = True
