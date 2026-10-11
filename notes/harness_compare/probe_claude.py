"""Check Claude Code against Tinker's Anthropic endpoint without a task sandbox."""

import json
import os
import subprocess
from pathlib import Path

from probe_anthropic import load_key


root = Path(__file__).resolve().parent
env = dict(os.environ)
env.update({
    "ANTHROPIC_API_KEY": load_key(),
    "ANTHROPIC_BASE_URL": "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
})
process = subprocess.run(
    [str(root / "claude-runtime/node_modules/.bin/claude"),
     "-p", "Reply READY only.",
     "--model", "thinkingmachines/Inkling-Small:peft:262144",
     "--tools", "",
     "--setting-sources", "",
     "--no-session-persistence",
     "--output-format", "json"],
    cwd=root / "claude-runtime", env=env, capture_output=True, text=True, timeout=120,
)
print(json.dumps({
    "exit_code": process.returncode,
    "stdout": process.stdout[:1200],
    "stderr": process.stderr[:1200],
}))
