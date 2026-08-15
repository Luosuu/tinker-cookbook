"""Mine Kokkos pull requests into raw SWE-compatible coding-RL candidates."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook.recipes.kokkos_rl.models import ChangedFile, KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.patching import (
    candidate_rejection_reasons,
    infer_era,
    linked_issue_numbers,
    partition_patch,
    sanitize_problem_statement,
)


class GitHubAPIError(RuntimeError):
    pass


class GitHubAPI:
    """Small async facade over GitHub's REST API using only the standard library."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    async def json(self, path: str, params: dict[str, object] | None = None) -> Any:
        return await asyncio.to_thread(self._request, path, params, "application/vnd.github+json")

    async def diff(self, path: str) -> str:
        value = await asyncio.to_thread(self._request, path, None, "application/vnd.github.v3.diff")
        if not isinstance(value, str):
            raise GitHubAPIError(f"Expected diff text from {path}")
        return value

    def _request(
        self, path: str, params: dict[str, object] | None, accept: str
    ) -> Any:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        request = urllib.request.Request(
            f"https://api.github.com/{path.lstrip('/')}{query}",
            headers={
                "Accept": accept,
                "User-Agent": "tinker-cookbook-kokkos-rl",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            raise GitHubAPIError(f"GitHub API {error.code} for {path}: {detail}") from error
        text = payload.decode()
        if accept.endswith("diff"):
            return text
        return json.loads(text)


@dataclass(frozen=True)
class RejectedPullRequest:
    pr_number: int
    title: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"pr_number": self.pr_number, "title": self.title, "reasons": self.reasons}


def _labels(pr: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["name"]) for item in pr.get("labels", []))


async def _problem_statement(
    api: GitHubAPI, repo: str, pr: dict[str, Any], issue_numbers: tuple[int, ...]
) -> str:
    issue_payloads = await asyncio.gather(
        *(api.json(f"repos/{repo}/issues/{number}") for number in issue_numbers)
    )
    issue_sections = []
    for issue in issue_payloads:
        if issue.get("pull_request"):
            continue
        title = str(issue.get("title", "")).strip()
        body = str(issue.get("body") or "").strip()
        issue_sections.append(f"{title}\n\n{body}".strip())
    if issue_sections:
        return "\n\n---\n\n".join(issue_sections)
    body = str(pr.get("body") or "").strip()
    return body or str(pr["title"])


async def _enrich_pr(
    api: GitHubAPI,
    repo: str,
    pr: dict[str, Any],
) -> KokkosInstance | RejectedPullRequest:
    number = int(pr["number"])
    title = str(pr["title"])
    file_payload = await api.json(f"repos/{repo}/pulls/{number}/files", {"per_page": 100})
    files = tuple(ChangedFile.from_api(item) for item in file_payload)
    reasons = candidate_rejection_reasons(
        files, title=title, description=str(pr.get("body") or "")
    )
    if reasons:
        return RejectedPullRequest(number, title, reasons)

    merge_commit = str(pr.get("merge_commit_sha") or "")
    if not merge_commit:
        return RejectedPullRequest(number, title, ("missing-merge-commit",))

    issue_numbers = linked_issue_numbers(str(pr.get("body") or ""))
    commit_payload, patch, raw_problem_statement = await asyncio.gather(
        api.json(f"repos/{repo}/commits/{merge_commit}"),
        api.diff(f"repos/{repo}/pulls/{number}"),
        _problem_statement(api, repo, pr, issue_numbers),
    )
    parents = commit_payload.get("parents", [])
    if not parents:
        return RejectedPullRequest(number, title, ("merge-commit-has-no-parent",))
    test_patch, code_patch = partition_patch(patch)
    if not test_patch.strip() or not code_patch.strip():
        return RejectedPullRequest(number, title, ("patch-partition-empty",))

    merged_at = str(pr["merged_at"])
    problem_statement = sanitize_problem_statement(raw_problem_statement)
    return KokkosInstance(
        instance_id=f"kokkos__kokkos-{number}",
        repo=repo,
        pr_number=number,
        title=title,
        problem_statement=problem_statement,
        merged_at=merged_at,
        base_commit=str(parents[0]["sha"]),
        merge_commit=merge_commit,
        era=infer_era(merged_at),
        labels=_labels(pr),
        changed_files=files,
        patch=patch,
        test_patch=test_patch,
        code_patch=code_patch,
        linked_issues=issue_numbers,
        metadata={"html_url": pr.get("html_url", "")},
    )


async def mine_candidates(
    api: GitHubAPI,
    *,
    repo: str = "kokkos/kokkos",
    base_branch: str = "develop",
    since: str = "2025-12-01",
    until: str | None = None,
    max_pages: int = 10,
    max_prs: int | None = None,
) -> tuple[list[KokkosInstance], list[RejectedPullRequest]]:
    merged: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = await api.json(
            f"repos/{repo}/pulls",
            {
                "state": "closed",
                "base": base_branch,
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not payload:
            break
        for pr in payload:
            merged_at = pr.get("merged_at")
            if not merged_at or str(merged_at)[:10] < since:
                continue
            if until and str(merged_at)[:10] > until:
                continue
            merged.append(pr)
            if max_prs is not None and len(merged) >= max_prs:
                break
        if max_prs is not None and len(merged) >= max_prs:
            break

    results = await asyncio.gather(*(_enrich_pr(api, repo, pr) for pr in merged))
    candidates = [item for item in results if isinstance(item, KokkosInstance)]
    rejected = [item for item in results if isinstance(item, RejectedPullRequest)]
    candidates.sort(key=lambda item: item.merged_at, reverse=True)
    rejected.sort(key=lambda item: item.pr_number, reverse=True)
    return candidates, rejected


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="kokkos/kokkos")
    parser.add_argument("--base-branch", default="develop")
    parser.add_argument("--since", default="2025-12-01")
    parser.add_argument("--until")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-prs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejections-output", type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    api = GitHubAPI(os.environ.get(args.token_env))
    candidates, rejected = await mine_candidates(
        api,
        repo=args.repo,
        base_branch=args.base_branch,
        since=args.since,
        until=args.until,
        max_pages=args.max_pages,
        max_prs=args.max_prs,
    )
    _write_jsonl(args.output, [item.to_dict() for item in candidates])
    if args.rejections_output:
        _write_jsonl(args.rejections_output, [item.to_dict() for item in rejected])
    print(f"wrote {len(candidates)} candidates; rejected {len(rejected)} pull requests")


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
