"""Smoke tests for the Nebius ConTree sandbox adapter."""

import os

import pytest
import pytest_asyncio

from tinker_cookbook.sandbox.contree_sandbox import (
    ContreeDockerfileSandboxFactory,
    ContreeSandbox,
    _dockerfile_instructions,
)

requires_contree = pytest.mark.skipif(
    not (os.environ.get("NEBIUS_SANDBOX_API_KEY") or os.environ.get("NEBIUS_API_KEY")),
    reason="Nebius ConTree is not configured",
)


def test_parse_harbor_dockerfile_subset(tmp_path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:24.04\n"
        "ENV DEBIAN_FRONTEND=noninteractive\n"
        "RUN echo first \\\n"
        "    && echo second\n"
        "WORKDIR /workspace/repo\n"
    )

    assert _dockerfile_instructions(dockerfile) == [
        ("FROM", "ubuntu:24.04"),
        ("ENV", "DEBIAN_FRONTEND=noninteractive"),
        ("RUN", "echo first && echo second"),
        ("WORKDIR", "/workspace/repo"),
    ]


def test_runtime_build_parallelism_is_validated(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ContreeDockerfileSandboxFactory(
            tmp_path / "cache.json", runtime_build_parallelism=0
        )


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    item = await ContreeSandbox.create(image="busybox:latest", timeout=120)
    yield item
    await item.cleanup()


@requires_contree
@pytest.mark.asyncio
async def test_persistent_commands_and_files(sandbox: ContreeSandbox) -> None:
    write = await sandbox.write_file("/tmp/test.sh", "#!/bin/sh\necho hello\n", executable=True)
    assert write.exit_code == 0, write.stderr

    run = await sandbox.run_command("/tmp/test.sh")
    assert run.exit_code == 0, run.stderr
    assert run.stdout.strip() == "hello"

    read = await sandbox.read_file("/tmp/test.sh", max_bytes=9)
    assert read.exit_code == 0, read.stderr
    assert read.stdout == "#!/bin/sh"
