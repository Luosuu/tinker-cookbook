"""Probe Claude Code's Tinker wire responses without a benchmark sandbox.

This uses a fake MCP shell and logs only response shape, never prompts or keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from diagnose_anthropic_turns import load_key
from completion_quality import classify_stream
from continuation import CONTINUE_PROMPT


UPSTREAM = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api"
MODEL = "thinkingmachines/Inkling-Small:peft:262144"


def summarize_response(body: bytes, content_type: str) -> dict:
    blocks = []
    stops = []
    if "event-stream" in content_type:
        for line in body.decode(errors="replace").splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_start":
                blocks.append(event.get("content_block", {}).get("type"))
            if event.get("type") == "message_delta":
                stops.append(event.get("delta", {}).get("stop_reason"))
        return {"blocks": blocks, "stops": stops}
    try:
        response = json.loads(body)
    except json.JSONDecodeError:
        return {"unparsed_bytes": len(body)}
    return {
        "blocks": [b.get("type") for b in response.get("content", [])],
        "stops": [response.get("stop_reason")],
    }


class ProxyHandler(BaseHTTPRequestHandler):
    request_count = 0
    effort: str | None = None

    def log_message(self, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        data = self.rfile.read(int(self.headers.get("content-length", "0")))
        try:
            request = json.loads(data)
            if self.effort and self.path.startswith("/v1/messages") and not self.path.startswith("/v1/messages/count_tokens"):
                request["output_config"] = {"effort": self.effort}
                data = json.dumps(request).encode()
            request_shape = {
                "model": request.get("model"),
                "max_tokens": request.get("max_tokens"),
                "stream": request.get("stream"),
                "thinking": request.get("thinking"),
                "output_config": request.get("output_config"),
                "tools": len(request.get("tools", [])),
                "messages": len(request.get("messages", [])),
            }
        except json.JSONDecodeError:
            request_shape = {"unparsed_bytes": len(data)}
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in {"host", "connection", "content-length", "accept-encoding"}
        }
        try:
            with httpx.Client(timeout=180) as client:
                response = client.post(UPSTREAM + self.path, content=data, headers=headers)
            response_body = response.content
            content_type = response.headers.get("content-type", "application/json")
            self.send_response(response.status_code)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            type(self).request_count += 1
            print(json.dumps({
                "api_request": type(self).request_count,
                "path": self.path,
                "request": request_shape,
                "status": response.status_code,
                "response": summarize_response(response_body, content_type),
            }), flush=True)
        except Exception as exc:
            self.send_error(502)
            print(json.dumps({"proxy_error": type(exc).__name__}), flush=True)


async def main() -> None:
    runtime = Path(sys.argv[1])
    key = load_key(Path(sys.argv[2]))
    instruction = Path(sys.argv[3]).read_text()
    ProxyHandler.effort = sys.argv[4] if len(sys.argv) > 4 else None
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="claude-wire-") as temp:
        temp_path = Path(temp)
        socket = str(temp_path / "mcp.sock")

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request = json.loads(await reader.readline())
            command = request.get("command", "")
            if command.startswith("ls "):
                output = "core\ncontainers\nalgorithms\nCMakeLists.txt\n"
            elif "RangePolicy" in command:
                output = "/workspace/repo/core/src/Kokkos_ExecPolicy.hpp:65:class RangePolicy\n"
            else:
                output = ""
            writer.write((json.dumps({"exit_code": 0, "stdout": output, "stderr": ""}) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=socket)
        mcp = temp_path / "mcp.json"
        mcp.write_text(json.dumps({"mcpServers": {"hpc": {
            "command": sys.executable,
            "args": [str(runtime.parent / "mcp_hpc.py")],
            "env": {"HPC_BRIDGE_SOCKET": socket},
        }}}))
        env = dict(os.environ)
        env.update({
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{proxy.server_port}",
            "ANTHROPIC_API_KEY": key,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "HOME": str(temp_path / "home"),
        })
        Path(env["HOME"]).mkdir()
        probe_resume = os.environ.get("HPC_RESUME_PROBE") == "1"
        force_resume = os.environ.get("HPC_FORCE_RESUME_PROBE") == "1"
        session_id = None
        for invocation in range(1, 3 if probe_resume else 2):
            args = [
                str(runtime / "node_modules/.bin/claude"),
                "-p", instruction if invocation == 1 else CONTINUE_PROMPT,
                "--model", MODEL,
                "--tools", "",
                "--mcp-config", str(mcp),
                "--strict-mcp-config",
                "--allowedTools=mcp__hpc__bash",
                "--permission-mode", "dontAsk",
                "--setting-sources", "",
                "--disable-slash-commands",
                "--max-turns", "12" if invocation == 1 else "2",
                "--output-format", "stream-json",
                "--verbose",
            ]
            if not probe_resume:
                args.append("--no-session-persistence")
            elif session_id:
                args.extend(("--resume", session_id))
            process = await asyncio.create_subprocess_exec(
                *args, cwd=runtime, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
            except TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
            events = []
            results = []
            lines = stdout.decode(errors="replace").splitlines()
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "assistant":
                    events.append([b.get("type") for b in event.get("message", {}).get("content", [])])
                elif event.get("type") == "result":
                    results.append(event)
                    events.append({"result_subtype": event.get("subtype"), "num_turns": event.get("num_turns"), "result_blank": not bool(event.get("result", "").strip())})
            quality = classify_stream(lines)
            print(json.dumps({
                "invocation": invocation,
                "cli_exit": process.returncode,
                "assistant_events": events,
                "completion_quality": quality,
                "has_session_id": bool(results and results[0].get("session_id")),
                "stderr_prefix": stderr.decode(errors="replace")[:180],
            }), flush=True)
            should_resume = (
                force_resume or
                (quality["kind"] == "thinking_only" and
                 quality["cli_result_subtype"] == "success" and
                 quality["cli_result_blank"])
            )
            if not (probe_resume and invocation == 1 and results and should_resume):
                break
            session_id = results[0].get("session_id")
            if not session_id:
                break
        server.close()
        await server.wait_closed()
        proxy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
