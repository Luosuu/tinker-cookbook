from unittest.mock import AsyncMock

import pytest

from .expansion import exclusion, scan
from .mine import RejectedPullRequest, _enrich_pr
from .validate import CommandResult


def row(number=1, patch="-old\n+new", **kwargs):
    return {
        "repo": "kokkos/kokkos",
        "instance_id": f"kokkos__kokkos-{number}",
        "pr_number": number,
        "code_patch": patch,
        **kwargs,
    }


def test_frozen_lineage_and_patch_normalization():
    frozen = [row()]
    assert (
        exclusion(row(2, "@@ -3,2 +8,2 @@\n- old\n+ new"), frozen, []) == "frozen-family:same-patch"
    )
    assert (
        exclusion(row(2, "-other\n+different", linked_issues=[3]), [row(linked_issues=[3])], [])
        == "frozen-family:same-issue"
    )
    assert (
        exclusion(row(2, "-other\n+different", title="Backport #1"), frozen, [])
        == "frozen-family:backport-reference"
    )
    assert exclusion(row(2), [], frozen) == "batch-duplicate:same-patch"
    assert exclusion(row(2, "-other\n+different"), frozen, []) is None


@pytest.mark.asyncio
async def test_resume_listing_does_not_claim_exhaustion_at_budget(tmp_path):
    api = AsyncMock()
    api.json.return_value = [{"number": n, "merged_at": None} for n in range(100)]
    result = await scan(
        api,
        tmp_path,
        repo="kokkos/kokkos",
        since="2020-01-01",
        until="2026-10-01",
        pages=1,
        enrich_limit=0,
        frozen=[],
    )
    assert result["listed"] == 100 and not result["scan_complete"]
    api.json.return_value = []
    result = await scan(
        api,
        tmp_path,
        repo="kokkos/kokkos",
        since="2020-01-01",
        until="2026-10-01",
        pages=1,
        enrich_limit=0,
        frozen=[],
    )
    assert api.json.call_args.args[1]["page"] == 2
    assert result["scan_complete"] and result["accepted_training"] == 0
    with pytest.raises(ValueError, match="changed"):
        await scan(
            api,
            tmp_path,
            repo="kokkos/kokkos",
            since="2021-01-01",
            until="2026-10-01",
            pages=1,
            enrich_limit=0,
            frozen=[],
        )


@pytest.mark.asyncio
async def test_rest_file_pagination_includes_second_page():
    api = AsyncMock()
    api.json.side_effect = [[{"filename": f"notes/{n}", "additions": 1} for n in range(100)], []]
    result = await _enrich_pr(api, "kokkos/kokkos", {"number": 1, "title": "fix"}, include_gpu=True)
    assert isinstance(result, RejectedPullRequest)
    assert api.json.call_count == 2
    assert api.json.call_args.args[1]["page"] == 2


@pytest.mark.parametrize("code", [-9, 124, 126, 127, 137, 143])
def test_infrastructure_exit_is_not_f2p(code):
    assert not CommandResult("test", "nonzero", code, 1, "", "").matched_expectation
