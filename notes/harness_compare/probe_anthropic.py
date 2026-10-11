"""Probe Tinker's Anthropic endpoint without exposing the credential."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def load_key() -> str:
    key = os.environ.get("TINKER_API_KEY")
    if key:
        return key
    path = Path(os.environ["TINKER_ENV_FILE"])
    for line in path.read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("TINKER_API_KEY missing")


def main() -> None:
    url = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1/messages"
    body = {
        "model": "thinkingmachines/Inkling-Small:peft:262144",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Reply with READY only."}],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": load_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
            print(json.dumps({
                "status": response.status,
                "model": data.get("model"),
                "content_types": [part.get("type") for part in data.get("content", [])],
                "text": [part.get("text") for part in data.get("content", []) if part.get("type") == "text"],
            }))
    except urllib.error.HTTPError as error:
        print(json.dumps({"status": error.code, "error": error.read(500).decode(errors="replace")}))


if __name__ == "__main__":
    main()
