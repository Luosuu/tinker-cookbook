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

from tinker_cookbook.recipes.kokkos_rl.dataset.ecosystem import (
    get_repository_profile,
    instance_id_for,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.models import ChangedFile, KokkosInstance
from tinker_cookbook.recipes.kokkos_rl.dataset.patching import (
    candidate_rejection_reasons,
    infer_accelerator,
    infer_era,
    linked_issue_numbers,
    partition_patch,
    sanitize_problem_statement,
)


class GitHubAPIError(RuntimeError):
    pass


class GitHubAPI:
    """Small async facade over GitHub's REST API using only the standard library."""

    def __init__(self, token: str | None = None, max_concurrency: int = 16) -> None:
        self._token = token
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def json(self, path: str, params: dict[str, object] | None = None) -> Any:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._request, path, params, "application/vnd.github+json"
            )

    async def diff(self, path: str) -> str:
        async with self._semaphore:
            value = await asyncio.to_thread(
                self._request, path, None, "application/vnd.github.v3.diff"
            )
        if not isinstance(value, str):
            raise GitHubAPIError(f"Expected diff text from {path}")
        return value

    async def graphql(self, query: str, variables: dict[str, object]) -> Any:
        """Run a GraphQL query, used to prefilter PRs without N REST calls per PR."""

        async with self._semaphore:
            return await asyncio.to_thread(self._graphql_request, query, variables)

    def _request(self, path: str, params: dict[str, object] | None, accept: str) -> Any:
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

    def _graphql_request(self, query: str, variables: dict[str, object]) -> Any:
        if not self._token:
            raise GitHubAPIError("GitHub GraphQL mining requires an authenticated token")
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "tinker-cookbook-kokkos-rl",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            raise GitHubAPIError(f"GitHub GraphQL API {error.code}: {detail}") from error
        if payload.get("errors"):
            raise GitHubAPIError(f"GitHub GraphQL errors: {payload['errors']}")
        return payload["data"]


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
    *,
    include_gpu: bool,
) -> KokkosInstance | RejectedPullRequest:
    number = int(pr["number"])
    title = str(pr["title"])
    file_payload = pr.get("_files")
    if file_payload is None:
        file_payload = await api.json(f"repos/{repo}/pulls/{number}/files", {"per_page": 100})
    files = tuple(ChangedFile.from_api(item) for item in file_payload)
    reasons = candidate_rejection_reasons(
        files,
        title=title,
        description=str(pr.get("body") or ""),
        repo=repo,
        include_gpu=include_gpu,
    )
    if reasons:
        return RejectedPullRequest(number, title, reasons)

    merge_commit = str(pr.get("merge_commit_sha") or "")
    if not merge_commit:
        return RejectedPullRequest(number, title, ("missing-merge-commit",))

    issue_numbers = linked_issue_numbers(str(pr.get("body") or ""), repo)
    base_commit = str(pr.get("_base_commit") or "")
    commit_payload: dict[str, Any] = {}
    try:
        if base_commit:
            patch, raw_problem_statement = await asyncio.gather(
                api.diff(f"repos/{repo}/pulls/{number}"),
                _problem_statement(api, repo, pr, issue_numbers),
            )
        else:
            commit_payload, patch, raw_problem_statement = await asyncio.gather(
                api.json(f"repos/{repo}/commits/{merge_commit}"),
                api.diff(f"repos/{repo}/pulls/{number}"),
                _problem_statement(api, repo, pr, issue_numbers),
            )
    except GitHubAPIError as error:
        if "diff exceeded the maximum number of files" in str(error):
            return RejectedPullRequest(number, title, ("diff-too-large",))
        raise
    if not base_commit:
        parents = commit_payload.get("parents", [])
        if not parents:
            return RejectedPullRequest(number, title, ("merge-commit-has-no-parent",))
        base_commit = str(parents[0]["sha"])
    test_patch, code_patch = partition_patch(patch, repo)
    if not test_patch.strip() or not code_patch.strip():
        return RejectedPullRequest(number, title, ("patch-partition-empty",))

    merged_at = str(pr["merged_at"])
    problem_statement = sanitize_problem_statement(raw_problem_statement)
    profile = get_repository_profile(repo)
    accelerator = infer_accelerator(files, title=title, description=str(pr.get("body") or ""))
    return KokkosInstance(
        instance_id=instance_id_for(repo, number),
        repo=repo,
        pr_number=number,
        title=title,
        problem_statement=problem_statement,
        merged_at=merged_at,
        base_commit=base_commit,
        merge_commit=merge_commit,
        era=infer_era(merged_at),
        labels=_labels(pr),
        changed_files=files,
        patch=patch,
        test_patch=test_patch,
        code_patch=code_patch,
        linked_issues=issue_numbers,
        configure_command=profile.configure_command,
        metadata={
            "html_url": pr.get("html_url", ""),
            "build_system": profile.build_system,
            "accelerator": accelerator or "cpu",
            **({"modal_gpu": "L4"} if accelerator == "cuda" else {}),
        },
    )


_GRAPHQL_PULL_REQUESTS = """
query PullRequests(
  $owner: String!, $name: String!, $base: String!, $cursor: String
) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: 100, after: $cursor, states: MERGED, baseRefName: $base,
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      nodes {
        number title body mergedAt url
        labels(first: 100) { nodes { name } }
        mergeCommit { oid parents(first: 1) { nodes { oid } } }
        files(first: 100) {
          totalCount
          nodes { path additions deletions changeType }
        }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


def _graphql_pr(node: dict[str, Any]) -> dict[str, Any]:
    merge_commit = node.get("mergeCommit") or {}
    parent_nodes = (merge_commit.get("parents") or {}).get("nodes") or []
    files_connection = node.get("files") or {}
    file_nodes = files_connection.get("nodes") or []
    status_names = {
        "ADDED": "added",
        "DELETED": "removed",
        "RENAMED": "renamed",
        "COPIED": "copied",
        "CHANGED": "changed",
        "MODIFIED": "modified",
    }
    return {
        "number": node["number"],
        "title": node["title"],
        "body": node.get("body") or "",
        "merged_at": node["mergedAt"],
        "merge_commit_sha": merge_commit.get("oid") or "",
        "html_url": node.get("url") or "",
        "labels": (node.get("labels") or {}).get("nodes") or [],
        "_base_commit": parent_nodes[0]["oid"] if parent_nodes else "",
        "_files": [
            {
                "filename": item["path"],
                "status": status_names.get(item.get("changeType", ""), "modified"),
                "additions": item.get("additions", 0),
                "deletions": item.get("deletions", 0),
            }
            for item in file_nodes
        ]
        + (
            [
                {
                    "filename": "__github_diff_too_large__",
                    "status": "modified",
                    "additions": 501,
                    "deletions": 0,
                }
            ]
            if int(files_connection.get("totalCount", 0)) > 100
            else []
        ),
    }


async def _list_merged_graphql(
    api: GitHubAPI,
    *,
    repo: str,
    base_branch: str,
    since: str,
    until: str | None,
    max_pages: int,
    max_prs: int | None,
) -> list[dict[str, Any]]:
    owner, name = repo.split("/", maxsplit=1)
    cursor: str | None = None
    merged: list[dict[str, Any]] = []
    for _ in range(max_pages):
        data = await api.graphql(
            _GRAPHQL_PULL_REQUESTS,
            {"owner": owner, "name": name, "base": base_branch, "cursor": cursor},
        )
        connection = data["repository"]["pullRequests"]
        for node in connection["nodes"]:
            merged_at = str(node.get("mergedAt") or "")
            if not merged_at or merged_at[:10] < since:
                continue
            if until and merged_at[:10] > until:
                continue
            merged.append(_graphql_pr(node))
            if max_prs is not None and len(merged) >= max_prs:
                return merged
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    return merged


async def mine_candidates(
    api: GitHubAPI,
    *,
    repo: str = "kokkos/kokkos",
    base_branch: str = "develop",
    since: str = "2025-12-01",
    until: str | None = None,
    max_pages: int = 10,
    max_prs: int | None = None,
    include_gpu: bool = True,
    transport: str = "rest",
) -> tuple[list[KokkosInstance], list[RejectedPullRequest]]:
    get_repository_profile(repo)
    if transport == "graphql":
        merged = await _list_merged_graphql(
            api,
            repo=repo,
            base_branch=base_branch,
            since=since,
            until=until,
            max_pages=max_pages,
            max_prs=max_prs,
        )
    else:
        merged = []
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

    results = await asyncio.gather(
        *(_enrich_pr(api, repo, pr, include_gpu=include_gpu) for pr in merged)
    )
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
    parser.add_argument("--base-branch")
    parser.add_argument("--since", default="2025-12-01")
    parser.add_argument("--until")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-prs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejections-output", type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--api-concurrency", type=int, default=16)
    parser.add_argument("--transport", choices=("rest", "graphql"), default="graphql")
    parser.add_argument(
        "--include-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain accelerator-specific candidates and label their required backend",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    api = GitHubAPI(os.environ.get(args.token_env), max_concurrency=args.api_concurrency)
    profile = get_repository_profile(args.repo)
    candidates, rejected = await mine_candidates(
        api,
        repo=args.repo,
        base_branch=args.base_branch or profile.default_branch,
        since=args.since,
        until=args.until,
        max_pages=args.max_pages,
        max_prs=args.max_prs,
        include_gpu=args.include_gpu,
        transport=args.transport,
    )
    _write_jsonl(args.output, [item.to_dict() for item in candidates])
    if args.rejections_output:
        _write_jsonl(args.rejections_output, [item.to_dict() for item in rejected])
    print(f"wrote {len(candidates)} candidates; rejected {len(rejected)} pull requests")


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
