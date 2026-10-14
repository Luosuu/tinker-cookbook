import pytest

from .preflight import gtest_names, selected_tests


def test_case_sensitive_inventory_prevents_false_nonempty_selection():
    names = gtest_names(
        "Running main()\nserial.\n  crs\n  count_fill\nTyped/0. # TypeParam = int\n  works\n"
    )
    assert names == ["Typed/0.works", "serial.count_fill", "serial.crs"]
    assert selected_tests(names, "*Crs*") == []
    assert selected_tests(names, "*crs*:*count_fill*") == ["serial.count_fill", "serial.crs"]
    assert selected_tests(names, "*-*.crs") == ["Typed/0.works", "serial.count_fill"]


async def _execute_shell(command):
    import asyncio

    from tinker_cookbook.sandbox import SandboxResult

    process = await asyncio.create_subprocess_exec(
        "bash", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return SandboxResult(
        stdout=stdout.decode(), stderr=stderr.decode(), exit_code=process.returncode
    )


@pytest.mark.asyncio
async def test_build_timeout_keeps_progress_and_exit_class():
    from .rivanna_validate import BuildEvidenceSandbox

    class ShellSandbox:
        async def run_command(self, command, **kwargs):
            return await _execute_shell(
                "cmake() { echo build-progress; sleep 2; }; export -f cmake; " + command
            )

    result = await BuildEvidenceSandbox(ShellSandbox()).run_command(
        "cmake --build build", timeout=1
    )
    assert result.exit_code == 124
    assert "build-progress" in result.stdout
    assert "TIME_LIMIT_SECONDS=1" in result.stdout


def test_annotation_cannot_submit_empty_baseline_selector():
    from dataclasses import replace

    from .models_test import _instance
    from .preflight import require_baseline_selection

    instance = replace(
        _instance(),
        metadata={"runtime_test_inventory": {"target": "RealTarget", "test_names": ["serial.crs"]}},
        p2p_commands=("./RealTarget --gtest_filter='*Crs*'",),
    )
    with pytest.raises(ValueError, match="no discovered"):
        require_baseline_selection(instance)
    require_baseline_selection(
        replace(instance, p2p_commands=("./RealTarget --gtest_filter='*crs*'",))
    )


@pytest.mark.asyncio
async def test_discovery_only_unsupported_input_never_falls_through_to_build():
    from dataclasses import replace
    from .models_test import _instance
    from .preflight import inventory

    result = await inventory(replace(_instance(), changed_files=()), discovery_only=True)
    assert result['stage'] == 'unsupported_discovery_input'
    assert not result['passed']
    assert result['target_builds'] == result['runtime_tests_executed'] == 0


def test_each_regression_command_is_checked_against_its_own_target():
    from dataclasses import replace

    import pytest

    from .models_test import _instance
    from .preflight import require_baseline_selection

    finder = '"$(find build -type f -name {} -perm -111 -print -quit)" --gtest_filter={}'
    inventory = {"targets": {
        "KokkosCore_UnitTest_Serial1": {"binary": "a", "test_names": ["serial.reduce"]},
        "KokkosAlgorithms_UnitTest": {"binary": "b", "test_names": ["serial.sort"]},
    }}

    def check(*commands):
        instance = replace(_instance(), p2p_commands=commands,
                           metadata={"runtime_test_inventory": inventory})
        require_baseline_selection(instance)

    check(finder.format("KokkosCore_UnitTest_Serial1", "serial.reduce"),
          finder.format("KokkosAlgorithms_UnitTest", "serial.sort"))
    with pytest.raises(ValueError, match="matches no"):
        check(finder.format("KokkosAlgorithms_UnitTest", "serial.reduce"))
    with pytest.raises(ValueError, match="not bound"):
        check(finder.format("KokkosCore_UnitTest_Serial2", "serial.reduce"))
