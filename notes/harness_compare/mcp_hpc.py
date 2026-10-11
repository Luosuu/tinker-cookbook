"""Expose exactly one sandbox-backed command tool to Claude Code over stdio MCP."""

from __future__ import annotations

import asyncio
import json
import os

from mcp.server.mcpserver import MCPServer


server = MCPServer("hpc")


@server.tool(name="bash", description=(
    "Run a shell command inside the isolated HPC task container. The working "
    "directory and files persist across calls. Use it to inspect, edit, build, "
    "and debug the repository."
))
async def bash(command: str) -> str:
    reader, writer = await asyncio.open_unix_connection(os.environ["HPC_BRIDGE_SOCKET"])
    try:
        writer.write((json.dumps({"command": command}) + "\n").encode())
        await writer.drain()
        response = json.loads(await reader.readline())
        if "error" in response:
            return f"Sandbox error: {response['error']}"
        return (
            f"exit_code={response['exit_code']}\n"
            f"stdout:\n{response['stdout']}\n"
            f"stderr:\n{response['stderr']}"
        )
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    server.run("stdio")
