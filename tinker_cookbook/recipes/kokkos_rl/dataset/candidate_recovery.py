"""Recover public source metadata and legacy attempt identities without network I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .patching import statement_source_evidence


def attempted_records(root: Path) -> dict[str, list[dict]]:
    """Receipts describe attempts; candidate queues and API prompts do not."""
    records: dict[str, list[dict]] = {}
    for pattern in ("**/receipt.json", "**/construction.json", "**/preflight-gates.json"):
        for path in sorted(root.glob(pattern)):
            value = json.loads(path.read_text())
            items = value if isinstance(value, list) else [value]
            for item in items:
                ident = item.get("instance_id")
                if not isinstance(ident, str):
                    continue
                records.setdefault(ident, []).append({
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": item.get("status", "attempted"),
                    "passed": item.get("passed"),
                    "stage": item.get("stage"),
                })
    return records


def recover_source_metadata(candidate: dict, sources: list[dict]) -> dict:
    """Missing cache is pending, never a claim about the actual public requirement."""
    result = json.loads(json.dumps(candidate))
    metadata = result.setdefault("metadata", {})
    original = metadata.get("statement_source_evidence")
    if isinstance(original, dict) and original.get("raw"):
        metadata["source_recovery"] = {"status": "existing_raw", "sources": sources}
        return result
    texts = [s["text"] for s in sources if isinstance(s.get("text"), str) and s["text"].strip()]
    if not texts:
        metadata["source_recovery"] = {"status": "pending_public_source", "sources": []}
        return result
    raw = "\n\n---\n\n".join(dict.fromkeys(texts))
    metadata["statement_source_evidence"] = statement_source_evidence(raw)
    metadata["source_recovery"] = {"status": "recovered_cached_public", "sources": sources}
    # Do not replace the original statement or infer requirements from patches.
    return result
