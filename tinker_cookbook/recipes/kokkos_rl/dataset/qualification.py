"""Versioned eligibility, public evidence validation, and fixed execution budgets.

No network or model calls. Old receipts are immutable inputs, never retry counters.
"""
from __future__ import annotations

import json
from pathlib import Path

from .expansion import digest
from .patching import statement_source_evidence


def reserve(path: Path, kind: str, key: str, limit: int, details: dict | None = None) -> None:
    """An exclusive reservation survives interruption; duplicates require reconciliation."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        ledger = json.loads(handle.read() or "{}")
        entries = ledger.setdefault(kind, [])
        if any(row["key"] == key for row in entries):
            raise ValueError("Reservation already exists; reconcile instead of replaying")
        if len(entries) >= limit:
            raise ValueError("Budget exhausted: " + kind)
        entries.append({"key": key, "details": details or {}})
        handle.seek(0)
        handle.truncate()
        json.dump(ledger, handle, indent=2)
        handle.flush()
        import os
        os.fsync(handle.fileno())


def qualification_key(instance, sources: list[dict], versions: dict) -> str:
    return digest({"instance_id": instance.instance_id, "base": instance.base_commit,
                   "test": digest(instance.test_patch), "code": digest(instance.code_patch),
                   "sources": digest(sources), "versions": versions})


def grant_requalification(path: Path, instance, *, old_key: str, new_key: str,
                          changes: list[dict], campaign: str) -> None:
    if old_key == new_key or not changes:
        raise ValueError("No material change; reuse prior decision")
    allowed = {"source", "extraction", "contract_coverage", "target_discovery", "toolchain", "verifier"}
    for change in changes:
        if (not change.get("before_sha256") or not change.get("after_sha256")
                or change.get("kind") not in allowed or not change.get("old_failure")
                or not change.get("causal_reason") or not change.get("evidence_sha256")
                or change.get("before_sha256") == change.get("after_sha256")):
            raise ValueError("Requalification lacks causal before/after evidence")
    # One grant per task in this frozen campaign, not one grant per arbitrary version bump.
    reserve(path, "qualification_grants", campaign + ":" + instance.instance_id, 3,
            {"old_key": old_key, "new_key": new_key, "changes": changes})


def writer_sources(sources: list[dict]) -> list[dict]:
    """Raw caches stay on disk; writers receive only reviewed public/base material."""
    result = []
    for source in sources:
        if source.get("comments_truncated") or source.get("pending_reference_ids"):
            raise ValueError("pending_public_source: bounded fetch left unresolved references/comments")
        if source.get("kind") == "base_api":
            if not source.get("base_commit") or not source.get("blob_sha"):
                raise ValueError("Base API source needs immutable commit and blob identity")
            text = source["text"]
        else:
            text = statement_source_evidence(source["text"])["sanitized"]
        result.append({"url": source["url"], "text": text, "sha256": digest(text),
                       "kind": source.get("kind", "public_discussion")})
    return result


def validate_support(items: list[dict], sources: list[dict]) -> None:
    by_url = {s["url"]: s["text"] for s in sources}
    if not items:
        raise ValueError("Public support is absent")
    for item in items:
        quote = item.get("quote")
        if (item.get("source_url") not in by_url or not isinstance(quote, str)
                or len(quote.strip()) < 8 or quote not in by_url[item["source_url"]]):
            raise ValueError("Public support is not an exact source excerpt")


def validate_contracts(contracts: list[dict], sources: list[dict]) -> None:
    seen = set()
    for contract in contracts:
        if not contract.get("id") or contract["id"] in seen or not contract.get("observable"):
            raise ValueError("Invalid or duplicate public contract")
        seen.add(contract["id"])
        validate_support(contract.get("support", []), sources)
        if contract.get("support_kind") == "repository_inference" and not contract.get("inference"):
            raise ValueError("Repository inference needs a public reasoning chain")


def audit_evidence_errors(audit, contracts: list[dict], sources: list[dict], statement: str) -> list[str]:
    findings = audit.contract_findings
    errors = []
    ids = [row.get("contract_id") for row in findings]
    required = {c["id"] for c in contracts}
    if len(ids) != len(set(ids)) or not required.issubset(ids):
        errors.append("Audit omits or duplicates public contracts")
    for row in findings:
        try:
            validate_support(row.get("support", []), sources)
        except ValueError as error:
            errors.append(str(error))
        if not row.get("observable") or row.get("classification") not in {"covered", "omission", "ambiguity", "overreach"}:
            errors.append("Audit finding lacks observable contract/classification")
        if row.get("classification") == "covered":
            quote = row.get("statement_quote", "")
            if len(quote) < 8 or quote not in statement:
                errors.append("Covered contract lacks statement evidence")
        elif not row.get("reason"):
            errors.append("Audit gap lacks explanation")
    if audit.assessment == "clear" and (audit.issues or any(r.get("classification") != "covered" for r in findings)):
        errors.append("Clear verdict contradicts evidence")
    if audit.assessment != "clear" and not any(r.get("classification") in {"omission", "ambiguity"} for r in findings):
        errors.append("Negative verdict has no supported observable gap")
    return errors


def execution_policy(instance) -> dict:
    policy = instance.metadata.get("execution_policy", {})
    result = {"command_seconds": 300, "build_seconds": 300, "image_seconds": 900,
              "grader_seconds": 600, "sandbox_seconds": 1500, **policy}
    for key in ("command_seconds", "build_seconds", "image_seconds", "grader_seconds"):
        if type(result[key]) is not int or not 1 <= result[key] <= 900:
            raise ValueError("Command budget must be frozen within 1..900 seconds")
    if not 900 <= result["sandbox_seconds"] <= 1500:
        raise ValueError("Sandbox budget exceeds fixed allocation")
    if result["build_seconds"] > 300 and not result.get("basis_sha256"):
        raise ValueError("Larger build tier needs preflight/timing evidence")
    return result


def decision_category(reason: str) -> tuple[str, bool]:
    value = reason.lower()
    if any(x in value for x in ("timeout", "timed out", "environment", "target_resolution", "discovery", "zero tests", "signal", "dependency")):
        return "engineering_blocked", True
    if any(x in value for x in ("source", "provenance", "contract", "statement", "audit")):
        return "source_or_generation_gap", True
    if "confirmed_maintenance_only" in value:
        return "genuinely_unsuitable", False
    return "not_yet_qualified", True
