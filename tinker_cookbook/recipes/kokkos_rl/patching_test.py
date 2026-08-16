from tinker_cookbook.recipes.kokkos_rl.models import ChangedFile
from tinker_cookbook.recipes.kokkos_rl.patching import (
    candidate_rejection_reasons,
    infer_era,
    is_forbidden_agent_path,
    linked_issue_numbers,
    partition_patch,
    sanitize_problem_statement,
    split_unified_diff,
)

PATCH = """diff --git a/core/src/Foo.hpp b/core/src/Foo.hpp
index 1111111..2222222 100644
--- a/core/src/Foo.hpp
+++ b/core/src/Foo.hpp
@@ -1 +1 @@
-old
+new
diff --git a/core/unit_test/TestFoo.cpp b/core/unit_test/TestFoo.cpp
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/core/unit_test/TestFoo.cpp
@@ -0,0 +1 @@
+TEST(foo, regression) {}
"""


def test_partition_patch_separates_hidden_tests() -> None:
    blocks = split_unified_diff(PATCH)
    assert [path for path, _ in blocks] == ["core/src/Foo.hpp", "core/unit_test/TestFoo.cpp"]
    test_patch, code_patch = partition_patch(PATCH)
    assert "TestFoo.cpp" in test_patch
    assert "Foo.hpp" not in test_patch
    assert "Foo.hpp" in code_patch
    assert "TestFoo.cpp" not in code_patch


def test_candidate_filter_requires_host_source_and_tests() -> None:
    good = (
        ChangedFile("core/src/Foo.hpp", "modified", 5, 2),
        ChangedFile("core/unit_test/TestFoo.cpp", "added", 10, 0),
    )
    assert candidate_rejection_reasons(good) == ()

    gpu_only = (
        ChangedFile("core/src/SYCL/Foo.hpp", "modified", 5, 2),
        ChangedFile("core/unit_test/TestFoo.cpp", "added", 10, 0),
    )
    assert candidate_rejection_reasons(gpu_only) == ("gpu-backend-only",)

    tests_only = (ChangedFile("core/unit_test/TestFoo.cpp", "added", 10, 0),)
    assert candidate_rejection_reasons(tests_only) == ("no-production-source-change",)

    assert candidate_rejection_reasons(good, title="Add NVIDIA Rubin support") == (
        "gpu-specific-change",
    )
    assert candidate_rejection_reasons(good, title="Final removal of deprecated API") == (
        "maintenance-cleanup",
    )


def test_ecosystem_profiles_classify_source_and_tests() -> None:
    kernels_files = (
        ChangedFile("sparse/src/KokkosSparse_spmv.hpp", "modified", 8, 2),
        ChangedFile("sparse/unit_test/Test_Sparse_spmv.hpp", "modified", 12, 0),
    )
    assert candidate_rejection_reasons(kernels_files, repo="kokkos/kokkos-kernels") == ()

    pykokkos_files = (
        ChangedFile("pykokkos/core/compiler.py", "modified", 8, 2),
        ChangedFile("tests/test_regressions.py", "modified", 12, 0),
    )
    assert candidate_rejection_reasons(pykokkos_files, repo="kokkos/pykokkos") == ()


def test_gpu_candidates_can_be_retained_for_modal_validation() -> None:
    files = (
        ChangedFile("core/src/Cuda/Kokkos_Cuda.hpp", "modified", 8, 2),
        ChangedFile("core/unit_test/cuda/TestCuda.cpp", "modified", 12, 0),
    )
    assert "gpu-backend-only" in candidate_rejection_reasons(files)
    assert candidate_rejection_reasons(files, include_gpu=True) == ()


def test_precommit_maintenance_is_rejected() -> None:
    files = (
        ChangedFile("common/tool.cpp", "modified", 5, 2),
        ChangedFile("tests/test_tool.cpp", "modified", 5, 2),
    )
    reasons = candidate_rejection_reasons(
        files,
        title="Add pre-commit check",
        repo="kokkos/kokkos-tools",
        include_gpu=True,
    )
    assert "maintenance-cleanup" in reasons


def test_issue_references_are_deduplicated_in_order() -> None:
    assert linked_issue_numbers("Fixes #12, resolves: #9 and closes #12") == (12, 9)
    assert linked_issue_numbers("Fixes https://github.com/kokkos/kokkos/issues/9413") == (9413,)


def test_forbidden_paths_cover_reward_hacking_surfaces() -> None:
    assert is_forbidden_agent_path("core/unit_test/TestFoo.cpp")
    assert is_forbidden_agent_path("core/unit_test/CMakeLists.txt")
    assert is_forbidden_agent_path(".github/workflows/ci.yml")
    assert not is_forbidden_agent_path("core/src/View/Kokkos_View.hpp")


def test_era_inference() -> None:
    assert infer_era("2026-08-14T00:00:00Z") == "cpp20"
    assert infer_era("2024-01-01T00:00:00Z") == "cpp17"
    assert infer_era("2022-01-01T00:00:00Z") == "cpp14"


def test_problem_statement_sanitizer_removes_patch_leakage() -> None:
    text = """Bug summary.

https://github.com/kokkos/kokkos/blob/abc/core/src/Foo.hpp#L10-L12

```c++
return exact_gold_fix();
```
"""
    sanitized = sanitize_problem_statement(text)
    assert "exact_gold_fix" not in sanitized
    assert "#L10" not in sanitized
    assert "Bug summary" in sanitized
