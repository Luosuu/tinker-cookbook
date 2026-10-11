import json
from dataclasses import replace

import pytest

from .audit_instructions import AuditResult
from .models_test import _instance
from .qualification import (reserve, grant_requalification, writer_sources, validate_contracts,
                            audit_evidence_errors, execution_policy, decision_category)


def test_atomic_budget_and_requalification_do_not_reset_on_version_churn(tmp_path):
    ledger=tmp_path/'ledger.json'
    reserve(ledger,'requests','a',1)
    with pytest.raises(ValueError,match='already'):reserve(ledger,'requests','a',1)
    with pytest.raises(ValueError,match='exhausted'):reserve(ledger,'requests','b',1)
    changes=[dict(kind='source',old_failure='missing issue',causal_reason='recovered explicit issue',evidence_sha256='a'*64,before_sha256='b'*64,after_sha256='c'*64)]
    with pytest.raises(ValueError,match='No material'):grant_requalification(ledger,_instance(),old_key='same',new_key='same',changes=changes,campaign='r')
    grant_requalification(ledger,_instance(),old_key='old',new_key='new',changes=changes,campaign='r')
    with pytest.raises(ValueError,match='already'):grant_requalification(ledger,_instance(),old_key='new',new_key='newer',changes=changes,campaign='r')
    assert len(json.loads(ledger.read_text())['qualification_grants'])==1


def test_writer_pack_preserves_short_reproducer_and_variant_but_excludes_solution():
    raw='This code segfaults:\n```c++\nKokkos::deep_copy(n,m); // alternative: n = Map{m.capacity()};\n```\nIf I hack the code like this:\n```patch\ndiff --git a/x b/x\n+SECRET_FIX\n```'
    sources=writer_sources([{'url':'public','text':raw,'extraction':{'raw':'PRIVATE_SENTINEL'}}])
    assert 'Map{m.capacity()}' in sources[0]['text']
    assert 'SECRET_FIX' not in str(sources) and 'PRIVATE_SENTINEL' not in str(sources)
    assert 'SECRET_FIX' in raw
    with pytest.raises(ValueError,match='immutable'):writer_sources([dict(url='base',text='API',kind='base_api')])


def test_repository_contract_inference_and_audit_require_evidence():
    sources=[dict(url='base',text='typedef MemorySpace memory_space;')]
    contracts=[dict(id='memory',observable='preserve selected memory space',support_kind='repository_inference',inference='Device exposes supplied MemorySpace',support=[dict(source_url='base',quote=sources[0]['text'])])]
    validate_contracts(contracts,sources)
    statement='Preserve the selected memory space when creating a ScatterView.'
    finding=dict(contract_id='memory',observable=contracts[0]['observable'],classification='covered',support=contracts[0]['support'],statement_quote=statement)
    audit=AuditResult('id','clear',(),statement,.99,'api_stop',(finding,))
    assert not audit_evidence_errors(audit,contracts,sources,statement)
    assert audit_evidence_errors(replace(audit,contract_findings=()),contracts,sources,statement)
    bad=dict(finding,support=[dict(source_url='base',quote='fabricated requirement')])
    assert audit_evidence_errors(replace(audit,contract_findings=(bad,)),contracts,sources,statement)
    over=dict(finding,classification='overreach',reason='private implementation detail')
    assert audit_evidence_errors(replace(audit,assessment='invalid',contract_findings=(over,)),contracts,sources,statement)


def test_resource_tiers_require_evidence_and_stay_bounded():
    instance=_instance()
    assert execution_policy(instance)['build_seconds']==300
    with pytest.raises(ValueError,match='evidence'):execution_policy(replace(instance,metadata={'execution_policy':{'build_seconds':600}}))
    assert execution_policy(replace(instance,metadata={'execution_policy':{'build_seconds':600,'basis_sha256':'a'*64}}))['build_seconds']==600
    with pytest.raises(ValueError,match='900'):execution_policy(replace(instance,metadata={'execution_policy':{'command_seconds':901}}))
    assert decision_category('target_resolution')[0]=='engineering_blocked'
    assert decision_category('missing public source')[0]=='source_or_generation_gap'
    assert decision_category('LLM says invalid')==('not_yet_qualified',True)


@pytest.mark.asyncio
async def test_missing_contract_variant_blocks_writer_result_without_private_feedback(tmp_path,monkeypatch):
    from unittest.mock import AsyncMock
    from . import revise_instructions as module
    source=dict(url='public',text='Empty maps crash during deep_copy after capacity-zero construction or rehash.')
    contract=dict(id='capacity_constructor',observable='capacity-zero constructor variant',support=[dict(source_url='public',quote=source['text'])])
    instance=replace(_instance(),code_patch='+PRIVATE_SENTINEL',metadata={'public_contracts':[contract]})
    monkeypatch.setattr(module,'public_sources',AsyncMock(return_value=[source]))
    finding=dict(contract_id=contract['id'],observable=contract['observable'],classification='omission',support=contract['support'],reason='Constructor variant missing')
    audit=AuditResult(instance.instance_id,'needs_revision',('missing constructor',),'A public statement about empty maps requiring a safe copy without segmentation faults.',.99,'api_stop',(finding,))
    completer=AsyncMock(return_value={'role':'assistant','content':json.dumps({'problem_statement':'Empty maps must allow deep_copy after rehash without crashing, preserving their observable empty state.','claims':[dict(text='No crash',source_url='public',quote=source['text'])]})})
    _,passed=await module.revise_and_audit(instance,completer=completer,api=AsyncMock(),reports_dir=tmp_path,initial_audit=audit)
    assert not passed and completer.call_count==1
    assert 'PRIVATE_SENTINEL' not in str(completer.call_args)
    assert 'variant omitted' in (tmp_path/'result.json').read_text()
