"""Signals are generated only in disposable child processes; core dumps disabled."""
import json
import shlex
import subprocess
import sys

import pytest

from .runtime_coverage import COVERAGE_FAILURE, classify_command_outcome, guarded_outcome
from .test_commands import guarded_command
from .validate import CommandResult

COMMAND = './case --gtest_filter=Suite.case'
STARTED = '[==========] Running 1 test from 1 test suite.\n[ RUN      ] Suite.case\n'


@pytest.mark.parametrize('code', [-6, -7, -8, -11])
def test_direct_target_signals_need_selected_active_case(code):
    outcome = classify_command_outcome(COMMAND, STARTED, code, True)
    assert outcome['status'] == 'failed'
    assert outcome['kind'] == 'target_program_signal'
    assert outcome['active_test'] == 'Suite.case'
    assert classify_command_outcome(COMMAND, STARTED, code, False)['status'] == 'error'
    assert classify_command_outcome(COMMAND, STARTED.replace('Suite.case', 'Suite.other'), code, True)['status'] == 'error'
    assert classify_command_outcome(COMMAND, STARTED + '[       OK ] Suite.case\n[  PASSED  ] 1 test.\n', code, True)['status'] == 'error'


@pytest.mark.parametrize('code', [-9, -15, -4, -1, 124, 126, 127, 137, 143, 245, 139])
def test_unknown_interruption_timeout_launcher_and_encoded_status_fail_closed(code):
    assert classify_command_outcome(COMMAND, STARTED, code, True)['status'] == 'error'


def test_resource_and_framework_evidence_override_active_test_signal():
    oom = classify_command_outcome(COMMAND, STARTED + 'Out of memory: Killed process', -11, True)
    assert oom['kind'] == 'resource_failure' and oom['status'] == 'error'
    allocation = classify_command_outcome(COMMAND, STARTED + 'terminate called after throwing std::bad_alloc', -6, True)
    assert allocation['kind'] == 'resource_failure' and allocation['status'] == 'error'
    before = classify_command_outcome(COMMAND, '[==========] Running 1 test from 1 test suite.', -11, True)
    assert before['kind'] == 'unknown_signal'
    missing = classify_command_outcome(COMMAND, 'Segmentation fault', -11, True)
    assert missing['kind'] == 'coverage_or_framework_failure'
    assert classify_command_outcome(COMMAND, STARTED+'Segmentation fault', 139, True)['status'] == 'failed'
    assert classify_command_outcome(COMMAND, STARTED+'timed out after 300 seconds', -11, True)['status'] == 'error'


def test_guard_raw_status_must_bind_command_and_transport():
    envelope = {'command': COMMAND, 'exit_code': -11, 'error': None}
    stdout = STARTED + 'KOKKOS_RUNTIME_COVERAGE ' + json.dumps(envelope) + '\n'
    for transport in [-11, 139, 245]:
        assert guarded_outcome(COMMAND, stdout, '', transport)['status'] == 'failed'
    assert guarded_outcome(COMMAND, stdout, '', 0)['status'] == 'error'
    assert guarded_outcome('different', stdout, '', 245)['status'] == 'error'
    assert guarded_outcome(COMMAND, STARTED, '', 245)['status'] == 'error'
    assert guarded_outcome(COMMAND, STARTED+'KOKKOS_RUNTIME_COVERAGE broken\n', '', 245)['status'] == 'error'


@pytest.mark.parametrize('signal_number,expected', [(11, 'failed'), (6, 'failed'), (9, 'error'), (15, 'error')])
def test_real_child_signal_is_consistent_direct_and_guarded(tmp_path, signal_number, expected):
    script = tmp_path / 'case.py'
    script.write_text('import os,signal,resource\nresource.setrlimit(resource.RLIMIT_CORE,(0,0))\n' + f'print({STARTED!r},flush=True)\nos.kill(os.getpid(),{signal_number})\n')
    command = f'{shlex.quote(sys.executable)} {shlex.quote(str(script))} --gtest_filter=Suite.case'
    direct = subprocess.run(['bash','-c','set -o pipefail; '+command], capture_output=True, text=True)
    outcome = classify_command_outcome(command, direct.stdout+'\n'+direct.stderr, direct.returncode, True)
    assert outcome['status'] == expected
    guarded = subprocess.run(['bash','-c',guarded_command(command)], capture_output=True, text=True)
    result = guarded_outcome(command, guarded.stdout, guarded.stderr, guarded.returncode)
    assert result['status'] == expected
    row = CommandResult(command, 'nonzero', guarded.returncode, 0, guarded.stdout, guarded.stderr)
    assert row.matched_expectation == (expected == 'failed')
    assert not CommandResult(command, 'zero', guarded.returncode, 0, guarded.stdout, guarded.stderr).matched_expectation
    if expected == 'error': assert guarded.returncode == COVERAGE_FAILURE
    assert not list(tmp_path.glob('core*'))


def test_real_timeout_is_not_a_program_regression(tmp_path):
    command = f'timeout 0.1s {shlex.quote(sys.executable)} -c "import time; time.sleep(2)"'
    completed = subprocess.run(['bash','-c',guarded_command(command)],capture_output=True,text=True)
    assert guarded_outcome(command,completed.stdout,completed.stderr,completed.returncode)['status']=='error'
    assert not CommandResult(command,'nonzero',completed.returncode,0,completed.stdout,completed.stderr).matched_expectation
