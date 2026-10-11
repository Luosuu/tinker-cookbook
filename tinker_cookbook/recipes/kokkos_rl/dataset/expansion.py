"""Bounded, resumable expansion queue. No task is training-ready without final evidence.

Frozen rows are used only for exclusion fingerprints, never passed to a model.
Each output directory is one immutable mining window; use a new one for fresh scans.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from .ecosystem import get_repository_profile
from .mine import GitHubAPI, RejectedPullRequest, _enrich_pr

VERSION = 1


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, values: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))
    temporary.replace(path)


def patch_lines(row: dict) -> set[str]:
    # Ignore hunk positions, whitespace and context so cherry-picks compare across bases.
    return {
        re.sub(r"\s+", "", line)
        for line in row.get("code_patch", "").splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")) and line[1:].strip()
    }


def related(candidate: dict, existing: dict) -> str | None:
    if candidate["repo"] != existing["repo"]:
        return None
    if candidate["instance_id"] == existing["instance_id"]:
        return "existing-id"
    if candidate.get("merge_commit") and candidate["merge_commit"] == existing.get("merge_commit"):
        return "same-merge"
    if set(candidate.get("linked_issues", [])) & set(existing.get("linked_issues", [])):
        return "same-issue"
    references = re.findall(
        r"(?:backport|cherry.pick)[\s\S]{0,100}?#(\d+)",
        candidate.get("title", "") + "\n" + candidate.get("problem_statement", ""),
        re.I,
    )
    if str(existing.get("pr_number")) in references:
        return "backport-reference"
    a, b = patch_lines(candidate), patch_lines(existing)
    if a and a == b:
        return "same-patch"
    if a and b and len(a & b) / min(len(a), len(b)) >= 0.85:
        return "near-patch"
    return None


def exclusion(candidate: dict, frozen: list[dict], accepted: list[dict]) -> str | None:
    # Conservatively isolate relations to ALL frozen tasks if no split is supplied.
    for row in frozen:
        if reason := related(candidate, row):
            return "frozen-family:" + reason
    for row in accepted:
        if reason := related(candidate, row):
            return "batch-duplicate:" + reason
    if re.search(r"backport|cherry.pick", candidate.get("title", ""), re.I):
        return "unresolved-backport"
    return None


async def scan(
    api: GitHubAPI,
    root: Path,
    *,
    repo: str,
    since: str,
    until: str,
    pages: int,
    enrich_limit: int,
    frozen: list[dict],
) -> dict:
    profile = get_repository_profile(repo)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "version": VERSION,
        "repo": repo,
        "since": since,
        "until": until,
        "frozen_sha256": digest(frozen),
        "miner_sha256": digest(
            {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("mine.py", "patching.py", "ecosystem.py", "models.py")
            }
        ),
        "queue_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Mining inputs/code changed; use a new output directory")
    atomic_json(config_path, config)
    state_path = root / "scan.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"next_page": 1, "complete": False, "listed": 0, "pending": [], "records": {}}
    )
    started = time.monotonic()
    # Creation ordering avoids skips when old PRs receive new comments during a scan.
    # Stop only on API exhaustion, not merged_at: an old PR can merge recently.
    for _ in range(pages):
        if state["complete"]:
            break
        payload = await api.json(
            f"repos/{repo}/pulls",
            {
                "state": "closed",
                "base": profile.default_branch,
                "sort": "created",
                "direction": "asc",
                "per_page": 100,
                "page": state["next_page"],
            },
        )
        state["listed"] += len(payload)
        for pr in payload:
            merged = str(pr.get("merged_at") or "")[:10]
            if merged and since <= merged <= until:
                state["pending"].append(pr)
        state["next_page"] += 1
        state["complete"] = len(payload) < 100
        atomic_json(state_path, state)
    candidates_path = root / "candidates.jsonl"
    candidates = rows(candidates_path) if candidates_path.exists() else []
    # Each successful enrichment is durable; API errors remain pending and retryable.
    for pr in state["pending"][:enrich_limit]:
        key = str(pr["number"])
        try:
            result = await _enrich_pr(api, repo, pr, include_gpu=True)
        except Exception as error:
            state["records"][key] = {
                "stage": "mining",
                "status": "environment_error",
                "reason": type(error).__name__,
            }
            atomic_json(state_path, state)
            break
        if isinstance(result, RejectedPullRequest):
            record = {"stage": "static_filter", "status": "excluded", "reasons": result.reasons}
        else:
            candidate = result.to_dict()
            # Replay after a crash between the two atomic writes is idempotent.
            candidates = [r for r in candidates if r["instance_id"] != candidate["instance_id"]]
            reason = exclusion(candidate, frozen, candidates)
            record = {
                "stage": "deduplication",
                "status": "quarantined" if reason else "candidate",
                "reason": reason,
                "input_sha256": digest(candidate),
            }
            if not reason:
                candidate["metadata"]["test_origin"] = "upstream"
                candidates.append(candidate)
                write_rows(candidates_path, candidates)
        state["records"][key] = record
        state["pending"] = [p for p in state["pending"] if p["number"] != pr["number"]]
        atomic_json(state_path, state)
    if not candidates_path.exists():
        write_rows(candidates_path, candidates)
    counts = Counter(r["status"] for r in state["records"].values())
    reasons = Counter(
        reason
        for r in state["records"].values()
        for reason in (r.get("reasons") or [r.get("reason")])
        if reason
    )
    report = {
        "repo": repo,
        "listed": state["listed"],
        "scan_complete": state["complete"],
        "pending_enrichment": len(state["pending"]),
        "processed": len(state["records"]),
        "status_counts": dict(counts),
        "reasons": dict(reasons),
        "unique_candidates": len(candidates),
        "accepted_training": 0,
        "backend": dict(Counter(r["metadata"].get("accelerator", "cpu") for r in candidates)),
        "modules": dict(
            Counter(f["filename"].split("/")[0] for r in candidates for f in r["changed_files"])
        ),
        "last_run_seconds": round(time.monotonic() - started, 2),
        "quality_boundary": "Candidates only; construction, audit and independent export evidence required.",
    }
    atomic_json(root / "report.json", report)
    write_rows(root / "accepted_training.jsonl", [])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", choices=["kokkos/kokkos", "kokkos/kokkos-kernels"], required=True)
    parser.add_argument("--since", default="2018-01-01")
    parser.add_argument("--until", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-instances", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--enrich-limit", type=int, default=8)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if not 1 <= args.pages <= 100 or not 0 <= args.enrich_limit <= 100:
        parser.error("pages must be 1..100 and enrich-limit 0..100 per invocation")
    report = asyncio.run(
        scan(
            GitHubAPI(os.environ.get(args.token_env), max_concurrency=2),
            args.output_dir,
            repo=args.repo,
            since=args.since,
            until=args.until,
            pages=args.pages,
            enrich_limit=args.enrich_limit,
            frozen=rows(args.frozen_instances),
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
