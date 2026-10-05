"""OpenAI-compatible proxy that makes the DSH harness budget-comparable.

DeepSeek Harness reaches Tinker through its own pi-ai provider, which can select a
reasoning level by *name* but cannot send Tinker's numeric ``reasoning_effort`` or the
``separate_reasoning`` flag. Sending neither silently changes inference parameters while
leaving model identity intact, which is precisely the comparability gap the prior
cross-harness study recorded.

This proxy closes that gap structurally: DSH points at it, and every upstream request
carries the reference inference parameters by construction rather than by configuration
that could drift. It also enforces the reference budgets, because ``dsh --profile
headless`` exposes no model-request cap and the reference baseline is defined at 40 turns
/ 80 tool calls / 65,536 sampled tokens.

Requests stream, so the response is passed through verbatim: only the request body is
rewritten.

Usage::

    python proxy.py --port 8801 --upstream https://.../oai/api/v1 --env-file key.env
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# The reference inference configuration, fixed by the v24 native baseline.
REFERENCE_REASONING_EFFORT = 0.9
REFERENCE_SEPARATE_REASONING = True


class Budget:
    """Per-episode request and token accounting, shared across proxy connections.

    One proxy instance serves one episode, so this is the episode's budget. Counting
    model requests here is what makes an episode's turn count match the reference even
    though the harness itself has no such cap.
    """

    def __init__(self, max_requests: int, max_input_tokens: int) -> None:
        self.max_requests = max_requests
        self.max_input_tokens = max_input_tokens
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def admit(self) -> tuple[bool, str]:
        with self._lock:
            if self.requests >= self.max_requests:
                return False, (
                    f"model-request budget exhausted ({self.requests}/{self.max_requests})"
                )
            if self.input_tokens >= self.max_input_tokens:
                return False, (
                    f"input-token budget exhausted ({self.input_tokens}/{self.max_input_tokens})"
                )
            self.requests += 1
            return True, ""

    def record_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        with self._lock:
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "max_requests": self.max_requests,
                "max_input_tokens": self.max_input_tokens,
            }


def load_env_file(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


def _inject_reference_inference(body: bytes) -> bytes:
    """Add Tinker's reasoning parameters, replacing anything the harness sent."""
    payload = json.loads(body or b"{}")
    payload["reasoning_effort"] = REFERENCE_REASONING_EFFORT
    payload["separate_reasoning"] = REFERENCE_SEPARATE_REASONING
    # The reference samples at temperature 1; the harness must not diverge here either.
    payload["temperature"] = 1
    # Token accounting is what enforces the reference input-token budget, so request usage
    # explicitly rather than depending on the caller to ask for it.
    if payload.get("stream"):
        options = payload.setdefault("stream_options", {})
        if isinstance(options, dict):
            options["include_usage"] = True
    return json.dumps(payload).encode()


def _join_upstream(base: str, path: str) -> str:
    """Join the configured upstream base with the request path without doubling a version.

    The DSH provider appends ``/chat/completions`` to a ``baseURL`` that already ends in
    ``/v1``, so the configured upstream is the API root. Callers may therefore pass either
    form; normalise to exactly one ``/v1``.
    """
    base = base.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1") :]
    return base + path


def _message_digest(messages: list[dict]) -> list[dict]:
    """Compact per-message view for prompt-equivalence auditing.

    Full bodies are large and contain the whole repository excerpt; this keeps the role,
    length, and a leading excerpt so the prompt can be compared against the reference
    without storing every byte twice.
    """
    out = []
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        out.append(
            {
                "role": message.get("role"),
                "chars": len(text or ""),
                "head": (text or "")[:160],
            }
        )
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ""
    upstream_key = ""
    budget = Budget(0, 0)
    transcript = None

    def log_message(self, fmt: str, *args: object) -> None:
        if os.environ.get("PROXY_VERBOSE"):
            super().log_message(fmt, *args)

    def _upstream_headers(self) -> dict[str, str]:
        """Headers for the upstream call.

        The endpoint sits behind Cloudflare, which rejects urllib's default
        ``Python-urllib/x.y`` User-Agent with a 403 HTML interstitial. A curl-like
        User-Agent is therefore required, not cosmetic.
        """
        return {
            "authorization": f"Bearer {self.upstream_key}",
            "content-type": self.headers.get("content-type", "application/json"),
            "accept": self.headers.get("accept", "*/*"),
            "user-agent": os.environ.get("PROXY_USER_AGENT", "curl/8.7.1"),
        }

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""

        if self.path.endswith("/models"):
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": os.environ.get("PROXY_MODEL", "model"),
                            "object": "model",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if "/chat/completions" in self.path:
            allowed, reason = self.budget.admit()
            if not allowed:
                # Ending the episode at the cap is the budget-matched behaviour; the
                # harness treats this as terminal and the driver still grades whatever
                # candidate the filesystem holds.
                payload = json.dumps(
                    {
                        "error": {
                            "message": reason,
                            "type": "budget_exhausted",
                            "code": "budget_exhausted",
                        }
                    }
                ).encode()
                self.send_response(429)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            body = _inject_reference_inference(body)
            if self.transcript is not None:
                self.transcript.write(
                    json.dumps(
                        {
                            "event": "request",
                            "budget": self.budget.snapshot(),
                            "request_sha256": hashlib.sha256(body).hexdigest(),
                            "message_roles": [
                                m.get("role") for m in json.loads(body).get("messages", [])
                            ][:4],
                            "tools": [
                                (t.get("function") or {}).get("name")
                                for t in json.loads(body).get("tools", [])
                            ],
                            "messages": _message_digest(json.loads(body).get("messages", [])),
                        }
                    )
                    + "\n"
                )
                self.transcript.flush()

        request = Request(
            _join_upstream(self.upstream, self.path),
            data=body if method == "POST" else None,
            method=method,
            headers=self._upstream_headers(),
        )
        try:
            upstream = urlopen(request)
        except HTTPError as error:
            detail = error.read()
            self.send_response(error.code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
            return
        except URLError as error:
            detail = json.dumps({"error": {"message": str(error)}}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
            return

        # Decide from the request, not only the response: Tinker's SSE reply carries no
        # content-type, so trusting the response header buffers a stream and loses its
        # usage frames.
        streaming = "text/event-stream" in (upstream.headers.get("content-type") or "")
        if not streaming and method == "POST" and "/chat/completions" in self.path:
            with contextlib.suppress(TypeError, ValueError):
                streaming = bool(json.loads(body).get("stream"))
        if streaming:
            self.send_response(upstream.status)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            # Usage can arrive on any chunk and the payload is unbounded, so parse each
            # SSE frame as it passes rather than scanning a trailing buffer.
            pending = b""
            for chunk in upstream:
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._observe_sse_line(line)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            payload = upstream.read()
            with contextlib.suppress(TypeError, ValueError):
                self.budget.record_usage(json.loads(payload).get("usage"))
            self.send_response(upstream.status)
            self.send_header(
                "content-type", upstream.headers.get("content-type", "application/json")
            )
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def _observe_sse_line(self, line: bytes) -> None:
        """Record usage from one SSE frame, if it carries any."""
        if not line.startswith(b"data:"):
            return
        chunk = line[5:].strip()
        if chunk in (b"", b"[DONE]"):
            return
        try:
            parsed = json.loads(chunk)
        except ValueError:
            return
        usage = parsed.get("usage")
        if not usage:
            return
        self.budget.record_usage(usage)
        if self.transcript is not None:
            self.transcript.write(
                json.dumps({"event": "usage", "usage": usage, "budget": self.budget.snapshot()})
                + "\n"
            )
            self.transcript.flush()

    def do_POST(self) -> None:
        self._relay("POST")

    def do_GET(self) -> None:
        self._relay("GET")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--max-input-tokens", type=int, default=5_000_000)
    parser.add_argument("--transcript", type=Path)
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise SystemExit("TINKER_API_KEY is not set")

    Handler.upstream = args.upstream
    Handler.upstream_key = key
    Handler.budget = Budget(args.max_requests, args.max_input_tokens)
    if args.transcript:
        Handler.transcript = args.transcript.open("a")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"event": "ready", "port": args.port, "upstream": args.upstream}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
