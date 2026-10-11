import json

from .candidate_recovery import attempted_records, recover_source_metadata


def test_legacy_root_receipt_and_preflight_are_attempts_not_candidate_lists(tmp_path):
    legacy = tmp_path / 'first-final-v10'
    legacy.mkdir()
    (legacy / 'receipt.json').write_text(json.dumps({'instance_id':'kokkos__kokkos-2739','status':'quarantined','stage':'instruction_audit'}))
    (tmp_path / 'preflight-gates.json').write_text(json.dumps([{'instance_id':'kokkos__kokkos-2841','passed':False}]))
    (tmp_path / 'candidates.json').write_text(json.dumps([{'instance_id':'untried'}]))
    found = attempted_records(tmp_path)
    assert set(found) == {'kokkos__kokkos-2739','kokkos__kokkos-2841'}
    assert found['kokkos__kokkos-2739'][0]['path'] == 'first-final-v10/receipt.json'
    assert len(found['kokkos__kokkos-2739'][0]['sha256']) == 64


def test_missing_public_cache_is_pending_and_recovery_keeps_statement():
    candidate = {'problem_statement':'legacy sanitized statement','metadata':{}}
    pending = recover_source_metadata(candidate, [])
    assert pending['metadata']['source_recovery']['status'] == 'pending_public_source'
    assert 'statement_source_evidence' not in pending['metadata']
    sources = [{'url':'https://github.com/org/repo/issues/1','text':'Public API fails with empty input.','cache_path':'public/issue.json'}]
    fixed = recover_source_metadata(candidate, sources)
    assert fixed['metadata']['statement_source_evidence']['raw'] == sources[0]['text']
    assert fixed['problem_statement'] == candidate['problem_statement']
    assert candidate['metadata'] == {}
    again = recover_source_metadata(fixed, [])
    assert again['metadata']['source_recovery']['status'] == 'existing_raw'
    assert again['metadata']['statement_source_evidence'] == fixed['metadata']['statement_source_evidence']
