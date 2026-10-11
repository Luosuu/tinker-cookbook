"""Small, credential-safe probe for Inkling tool-use termination on Tinker."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx


URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1/messages"
MODEL = "thinkingmachines/Inkling-Small:peft:262144"


def load_key(path: Path) -> str:
    key = os.environ.get("TINKER_API_KEY")
    if key:
        return key
    for line in path.read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("TINKER_API_KEY missing")


def request(key: str, messages: list[dict], effort: str | None) -> dict:
    body: dict = {
        "model": MODEL,
        "max_tokens": 1024,
        "temperature": 1,
        "system": "You are a software engineer. Use the bash tool to inspect the repository, then solve the requested issue.",
        "messages": messages,
        "tools": [{
            "name": "bash",
            "description": "Run a shell command inside the isolated repository.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }],
    }
    if effort:
        body["output_config"] = {"effort": effort}
    with httpx.Client(timeout=120) as client:
        response = client.post(
            URL,
            json=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
    if response.status_code != 200:
        detail = response.text[:500].replace(key, "<redacted>")
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def main() -> None:
    key = load_key(Path(sys.argv[1]))
    for effort in (None, "high"):
        messages = [{"role": "user", "content": "Find the RangePolicy definition in /workspace/repo. First inspect the repository with bash, then explain what you found."}]
        for turn in range(1, 5):
            response = request(key, messages, effort)
            blocks = response.get("content", [])
            print(json.dumps({
                "effort": effort or "default",
                "turn": turn,
                "stop_reason": response.get("stop_reason"),
                "block_types": [block.get("type") for block in blocks],
                "thinking_prefix": next((block.get("thinking", "")[:110] for block in blocks if block.get("type") == "thinking"), None),
                "text_prefix": next((block.get("text", "")[:110] for block in blocks if block.get("type") == "text"), None),
                "output_tokens": response.get("usage", {}).get("output_tokens"),
            }), flush=True)
            messages.append({"role": "assistant", "content": blocks})
            tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
            if not tool_uses:
                break
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block["id"], "content": "exit_code=0\nstdout:\n/workspace/repo/core/src/Kokkos_ExecPolicy.hpp:65:class RangePolicy\nstderr:\n"}
                for block in tool_uses
            ]})


if __name__ == "__main__":
    main()
