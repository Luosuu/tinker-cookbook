from tinker_cookbook.recipes.kokkos_rl.models import ChangedFile, KokkosInstance


def _instance() -> KokkosInstance:
    return KokkosInstance(
        instance_id="kokkos__kokkos-1",
        repo="kokkos/kokkos",
        pr_number=1,
        title="Fix a bug",
        problem_statement="A focused problem.",
        merged_at="2026-08-14T00:00:00Z",
        base_commit="a" * 40,
        merge_commit="b" * 40,
        era="cpp20",
        labels=("Bug",),
        changed_files=(ChangedFile("core/src/Foo.hpp", "modified", 3, 2),),
        patch="full",
        test_patch="tests",
        code_patch="code",
        build_targets=("Kokkos_CoreUnitTest_Serial",),
        fail_to_pass=("Foo.Regression",),
        pass_to_pass=("Foo.Existing",),
        f2p_commands=(
            "./build/core/unit_test/Kokkos_CoreUnitTest_Serial --gtest_filter=Foo.Regression",
        ),
        p2p_commands=(
            "./build/core/unit_test/Kokkos_CoreUnitTest_Serial --gtest_filter=Foo.Existing",
        ),
    )


def test_round_trip_preserves_swe_compatibility_fields() -> None:
    original = _instance()
    value = original.to_dict()
    assert value["FAIL_TO_PASS"] == ["Foo.Regression"]
    assert value["PASS_TO_PASS"] == ["Foo.Existing"]
    assert value["hints_text"] == ""
    assert value["created_at"] == original.merged_at
    assert value["version"] == original.era
    assert KokkosInstance.from_dict(value) == original


def test_validation_ready_requires_verification_metadata() -> None:
    assert _instance().is_validation_ready


def test_compile_failure_instance_does_not_need_a_runtime_f2p_command() -> None:
    value = _instance().to_dict()
    value["f2p_commands"] = []
    value["metadata"] = {"f2p_stage": "build"}
    assert KokkosInstance.from_dict(value).is_validation_ready
