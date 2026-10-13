import json
import shutil

import pytest

from .target_discovery import dependency_arguments, discover, resolve_ownership


def test_ambiguity_failures_and_missing_dependencies_are_not_zero_tests():
    rows = [{'target':'A','changed_dependencies':['test.hpp']}, {'target':'B','changed_dependencies':['test.hpp']}]
    assert resolve_ownership(['test.hpp'], rows, [])['target'] is None
    assert resolve_ownership(['new.hpp'], rows, [])['unresolved'] == ['new.hpp']
    assert resolve_ownership(['test.hpp'], rows[:1], [{'error':'timeout'}])['target'] is None
    assert resolve_ownership(['test.hpp'], rows[:1], [])['target'] == 'A'
    args=dependency_arguments(['c++','-Iinc','-c','file.cpp','-o','file.o','-MD','-MF','file.d'],'new.d')
    assert '-c' not in args and 'file.o' not in args and 'file.d' not in args
    assert args[-5:] == ['-M','-MF','new.d','-MT','discovery']


def test_actual_preprocessor_follows_transitive_and_conditional_includes(tmp_path):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler required on execution host')
    (tmp_path/'active.hpp').write_text('// active\n')
    (tmp_path/'inactive.hpp').write_text('// inactive\n')
    (tmp_path/'wrapper.hpp').write_text('#include "active.hpp"\n#if 0\n#include "inactive.hpp"\n#endif\n')
    (tmp_path/'test.cpp').write_text('#include "wrapper.hpp"\n')
    reply=tmp_path/'build/.cmake/api/v1/reply';reply.mkdir(parents=True)
    model={'name':'RealSerial','type':'EXECUTABLE','sources':[{'path':'test.cpp'}]}
    (reply/'target-real.json').write_text(json.dumps(model))
    command={'directory':str(tmp_path),'file':'test.cpp','arguments':[compiler,'-I.','-c','test.cpp','-o','build/CMakeFiles/RealSerial.dir/test.cpp.o']}
    (tmp_path/'build/compile_commands.json').write_text(json.dumps([command]))
    found=discover(tmp_path,['active.hpp'])
    assert found['target']=='RealSerial'
    assert found['observations'][0]['translation_unit']=='test.cpp'
    assert discover(tmp_path,['inactive.hpp'])['target'] is None
    assert not list(tmp_path.rglob('*.o'))


def test_host_targets_only_for_files_no_serial_target_owns(tmp_path):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler required on execution host')
    targets = {
        'KokkosCore_UnitTest_Serial1': ('serial.cpp', 'shared.hpp'),
        'KokkosCore_UnitTest_OpenMP': ('omp.cpp', 'shared.hpp'),
        'KokkosCore_IncrementalTest_SERIAL': ('inc_s.cpp', 'inc.hpp'),
        'KokkosCore_IncrementalTest_OPENMP': ('inc_o.cpp', 'inc.hpp'),
        'KokkosCore_PerformanceTest_Atomic': ('perf.cpp', 'alg.hpp'),
        'KokkosAlgorithms_UnitTest': ('alg.cpp', 'alg.hpp'),
    }
    reply = tmp_path/'build/.cmake/api/v1/reply'; reply.mkdir(parents=True)
    commands = []
    for name, (source, header) in targets.items():
        (tmp_path/header).write_text('// header\n')
        (tmp_path/source).write_text(f'#include "{header}"\n')
        (reply/f'target-{name}.json').write_text(json.dumps({'name': name, 'type': 'EXECUTABLE', 'sources': [{'path': source}]}))
        commands.append({'directory': str(tmp_path), 'file': source, 'arguments': [compiler, '-I.', '-c', source, '-o', f'build/CMakeFiles/{name}.dir/{source}.o']})
    (tmp_path/'build/compile_commands.json').write_text(json.dumps(commands))
    shared = discover(tmp_path, ['shared.hpp'])
    assert shared['target'] == 'KokkosCore_UnitTest_Serial1'
    assert {row['target'] for row in shared['observations']} == {'KokkosCore_UnitTest_Serial1'}
    assert discover(tmp_path, ['inc.hpp'])['target'] == 'KokkosCore_IncrementalTest_SERIAL'
    assert discover(tmp_path, ['alg.hpp'])['target'] == 'KokkosAlgorithms_UnitTest'
    assert discover(tmp_path, ['shared.hpp', 'alg.hpp'])['target'] is None


def test_cuda_tasks_resolve_to_cuda_executables(tmp_path):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler required on execution host')
    targets = {
        'Kokkos_CoreUnitTest_Serial1': 'serial.cpp',
        'Kokkos_CoreUnitTest_Cuda1': 'cuda.cpp',
        'Kokkos_CoreUnitTest_HIP': 'hip.cpp',
    }
    reply = tmp_path/'build/.cmake/api/v1/reply'; reply.mkdir(parents=True)
    (tmp_path/'shared.hpp').write_text('// header\n')
    commands = []
    for name, source in targets.items():
        (tmp_path/source).write_text('#include "shared.hpp"\n')
        (reply/f'target-{name}.json').write_text(json.dumps({'name': name, 'type': 'EXECUTABLE', 'sources': [{'path': source}]}))
        commands.append({'directory': str(tmp_path), 'file': source, 'arguments': [compiler, '-I.', '-c', source, '-o', f'build/CMakeFiles/{name}.dir/{source}.o']})
    (tmp_path/'build/compile_commands.json').write_text(json.dumps(commands))
    assert discover(tmp_path, ['shared.hpp'])['target'] == 'Kokkos_CoreUnitTest_Serial1'
    assert discover(tmp_path, ['shared.hpp'], toolchain='cuda')['target'] == 'Kokkos_CoreUnitTest_Cuda1'


def test_nvcc_wrapper_units_are_preprocessed_in_parallel(tmp_path):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler required on execution host')
    wrapper = tmp_path/'bin/nvcc_wrapper'; wrapper.parent.mkdir()
    # Like nvcc_wrapper, refuse -M; preprocessing still works.
    wrapper.write_text(f'#!/bin/sh\ncase " $* " in *" -M "*) exit 0;; esac\nexec {compiler} "$@"\n')
    wrapper.chmod(0o755)
    reply = tmp_path/'build/.cmake/api/v1/reply'; reply.mkdir(parents=True)
    (tmp_path/'shared.hpp').write_text('// header\n')
    commands = []
    for i in range(4):
        name, source = f'Kokkos_CoreUnitTest_Cuda{i}', f'cuda{i}.cpp'
        (tmp_path/source).write_text('#include "shared.hpp"\n' if i == 2 else '\n')
        (reply/f'target-{name}.json').write_text(json.dumps({'name': name, 'type': 'EXECUTABLE', 'sources': [{'path': source}]}))
        commands.append({'directory': str(tmp_path), 'file': source, 'arguments': [str(wrapper), '-I.', '-c', source, '-o', f'build/CMakeFiles/{name}.dir/{source}.o']})
    (tmp_path/'build/compile_commands.json').write_text(json.dumps(commands))
    found = discover(tmp_path, ['shared.hpp'], toolchain='cuda')
    assert found['target'] == 'Kokkos_CoreUnitTest_Cuda2' and not found['failures']
    assert len(found['observations']) == 4


def test_other_backend_test_directories_are_not_built():
    from .target_discovery import built_for

    assert built_for('core/unit_test/TestReducers.hpp', 'cuda')
    assert built_for('core/unit_test/cuda/TestCuda_Init.cpp', 'cuda')
    assert not built_for('core/unit_test/hip/TestHIP_AsyncLauncher.cpp', 'cuda')
    assert not built_for('core/unit_test/openmp/TestOpenMP.hpp', 'cuda')
    assert built_for('core/unit_test/openmp/TestOpenMP.hpp', 'cpu')
    assert not built_for('core/unit_test/cuda/TestCuda_Init.cpp', 'cpu')
