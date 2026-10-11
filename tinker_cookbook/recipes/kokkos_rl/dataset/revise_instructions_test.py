import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from .audit_instructions import AuditResult
from .models_test import _instance
from .revise_instructions import revise_and_audit, validate_revision

SOURCE = {
    "url": "https://github.com/example/repo/issues/1",
    "text": "Empty maps with zero capacity crash during deep_copy after rehash(0).",
}
STATEMENT = "Kokkos::UnorderedMap must support copying empty maps with zero capacity after rehash(0), without crashing or corrupting the destination."


def draft():
    return {
        "problem_statement": STATEMENT,
        "claims": [
            {
                "text": "Zero capacity copying must not crash",
                "source_url": SOURCE["url"],
                "quote": SOURCE["text"],
            }
        ],
    }


def test_revision_rejects_fabricated_public_provenance():
    assert validate_revision(draft(), [SOURCE]) == STATEMENT
    value = draft()
    value["claims"][0]["quote"] = "not in original source"
    with pytest.raises(ValueError, match="exact public"):
        validate_revision(value, [SOURCE])


@pytest.mark.asyncio
async def test_author_never_sees_private_patch_and_revision_is_reaudited(tmp_path, monkeypatch):
    from . import revise_instructions as module

    instance = replace(_instance(), code_patch="+PRIVATE_IMPLEMENTATION_MUST_NOT_REACH_AUTHOR")
    monkeypatch.setattr(module, "public_sources", AsyncMock(return_value=[SOURCE]))
    audit = AuditResult(instance.instance_id, "clear", (), STATEMENT, 0.99, "api_stop")
    auditor = AsyncMock(return_value=audit)
    monkeypatch.setattr(module, "_audit_one", auditor)
    completer = AsyncMock(return_value={"role": "assistant", "content": json.dumps(draft())})
    initial = replace(audit, assessment="needs_revision", issues=("missing public behavior",))
    revised, passed = await revise_and_audit(
        instance, completer=completer, api=AsyncMock(), reports_dir=tmp_path, initial_audit=initial
    )
    assert passed and revised.problem_statement == STATEMENT
    assert revised.code_patch == instance.code_patch and revised.test_patch == instance.test_patch
    assert "PRIVATE_IMPLEMENTATION" not in str(completer.call_args)
    assert auditor.call_count == 1
    assert json.loads((tmp_path / "result.json").read_text())["revisions"] == 1
    # A completed resume does not issue more paid calls.
    await revise_and_audit(
        instance, completer=completer, api=AsyncMock(), reports_dir=tmp_path, initial_audit=initial
    )
    assert completer.call_count == 1 and auditor.call_count == 1


@pytest.mark.asyncio
async def test_revision_stops_at_two_rounds(tmp_path, monkeypatch):
    from . import revise_instructions as module

    instance = _instance()
    monkeypatch.setattr(module, "public_sources", AsyncMock(return_value=[SOURCE]))
    audit = AuditResult(
        instance.instance_id, "needs_revision", ("uncertain",), STATEMENT, 0.7, "api_stop"
    )
    auditor = AsyncMock(return_value=audit)
    monkeypatch.setattr(module, "_audit_one", auditor)
    completer = AsyncMock(return_value={"role": "assistant", "content": json.dumps(draft())})
    _, passed = await revise_and_audit(
        instance, completer=completer, api=AsyncMock(), reports_dir=tmp_path
    )
    assert not passed and completer.call_count == 2 and auditor.call_count == 3


@pytest.mark.asyncio
async def test_interrupted_revision_is_not_replayed(tmp_path, monkeypatch):
    from . import revise_instructions as module

    instance = _instance()
    monkeypatch.setattr(module, "public_sources", AsyncMock(return_value=[SOURCE]))
    audit = AuditResult(
        instance.instance_id, "needs_revision", ("missing",), STATEMENT, 0.95, "api_stop"
    )
    (tmp_path / "revision-1.pending.json").write_text("{}")
    completer = AsyncMock()
    with pytest.raises(ValueError, match="no automatic replay"):
        await revise_and_audit(
            instance,
            completer=completer,
            api=AsyncMock(),
            reports_dir=tmp_path,
            initial_audit=audit,
        )
    completer.assert_not_called()


@pytest.mark.asyncio
async def test_public_extraction_includes_title_and_nonclosing_issue_reference():
    from .revise_instructions import public_sources

    api = AsyncMock()
    api.json.side_effect = [
        dict(
            title="Add useful API",
            body="Address issue #42",
            html_url="https://github.com/example/repo/pull/1",
        ),
        dict(
            title="API failure",
            body="Observable issue behavior.",
            html_url="https://github.com/example/repo/issues/42",
        ),
    ]
    result = await public_sources(replace(_instance(), linked_issues=()), api)
    assert len(result) == 2
    assert result[0]["text"].startswith("Add useful API")
    assert result[1]["extraction"]["raw"] == result[1]["text"]


@pytest.mark.asyncio
async def test_public_reference_traversal_includes_title_comments_and_linked_pr_issue():
    from dataclasses import replace
    from .models_test import _instance
    from .revise_instructions import public_sources

    class API:
        async def json(self, path, params=None):
            if '/pulls/' in path:
                return {'html_url':'https://github.com/kokkos/kokkos/pull/1','title':'Issue 2401: operation','body':'Alternative to #2829','comments':2}
            if path.endswith('/comments'):
                return [{'html_url':'https://github.com/kokkos/kokkos/pull/1#issuecomment-1','body':'Addresses issue #2826','user':{'type':'User'}}, {'html_url':'https://github.com/kokkos/kokkos/pull/1#issuecomment-2','body':'Issue #9999','user':{'type':'Bot'}}]
            number=int(path.rsplit('/',1)[-1])
            return {'html_url':f'https://github.com/kokkos/kokkos/issues/{number}','title':f'Issue {number}','body':'Issue #1936' if number==2829 else 'Public reproduction and expected behavior.'}
    sources=await public_sources(replace(_instance(),pr_number=1,linked_issues=()),API())
    urls=[s['url'] for s in sources]
    assert any(u.endswith('/2401') for u in urls)
    assert any(u.endswith('/2826') for u in urls)
    assert any(u.endswith('/1936') for u in urls)
    assert not any(u.endswith('/9999') for u in urls)
