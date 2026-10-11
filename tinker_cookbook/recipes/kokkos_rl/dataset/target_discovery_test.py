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
