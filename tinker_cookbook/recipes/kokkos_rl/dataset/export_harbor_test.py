import json
import tomllib
from dataclasses import replace

from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import (
    _agent_allowed_code_patch,
    export_instance,
)
from tinker_cookbook.recipes.kokkos_rl.dataset.models_test import _instance


def test_export_hides_tests_and_adds_clean_room_guard(tmp_path) -> None:
    instance = _instance()
    task_dir = export_instance(instance, tmp_path)

    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert "ca-certificates" in (task_dir / "environment" / "Dockerfile").read_text()
    assert (task_dir / "tests" / "test.patch").read_text() == "tests"
    assert (task_dir / "solution" / "gold.patch").read_text() == "code"
    assert "git apply" in (task_dir / "solution" / "solve.sh").read_text()
    assert "Foo.Regression" not in (task_dir / "instruction.md").read_text()
    instruction = (task_dir / "instruction.md").read_text()
    assert "Network access is unavailable" in instruction
    assert "git history contains" in instruction
    assert "only the base revision" in instruction
    assert "/tests/test.sh" not in instruction

    test_script = (task_dir / "tests" / "test.sh").read_text()
    assert 'git --no-replace-objects diff --no-ext-diff --name-only "$baseline"' in test_script
    assert "git ls-files --others" in test_script
    assert r"sed '\#^build/#d'" in test_script
    assert "git apply --whitespace=nowarn /tests/test.patch" in test_script
    assert "echo 1" in test_script

    metadata = json.loads((task_dir / "metadata.json").read_text())
    assert metadata["instance_id"] == instance.instance_id

    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    assert task_config["schema_version"] == "1.4"
    assert task_config["task"]["name"] == f"swe-kokkos/{instance.instance_id}"
    assert task_config["environment"]["network_mode"] == "public"
    assert task_config["agent"]["network_mode"] == "allowlist"
    assert task_config["agent"]["allowed_hosts"] == ["api.openai.com"]
    assert task_config["verifier"]["network_mode"] == "no-network"


def test_oracle_patch_excludes_verifier_protected_paths() -> None:
    instance = replace(
        _instance(),
        code_patch="""diff --git a/core/src/Foo.hpp b/core/src/Foo.hpp
--- a/core/src/Foo.hpp
+++ b/core/src/Foo.hpp
@@ -1 +1 @@
-old
+new
diff --git a/example/CMakeLists.txt b/example/CMakeLists.txt
--- a/example/CMakeLists.txt
+++ b/example/CMakeLists.txt
@@ -1 +1 @@
-old
+new
""",
    )

    allowed = _agent_allowed_code_patch(instance)

    assert "core/src/Foo.hpp" in allowed
    assert "CMakeLists.txt" not in allowed


# Exercise the emitted shell scripts against real git repositories, including
# commits and replacement refs, without cloud services or a C++ toolchain.
def _git(repo, *args):
    import subprocess

    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _local_instance(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "src.txt").write_text("broken")
    (repo / "CMakeLists.txt").write_text("original")
    (repo / "tests").mkdir()
    (repo / "tests" / "check.txt").write_text("original\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "tests" / "check.txt").write_text("hidden\n")
    test_patch = _git(repo, "diff") + "\n"
    _git(repo, "checkout", "--", "tests/check.txt")
    instance = replace(
        _instance(),
        base_commit=base,
        test_patch=test_patch,
        build_command="true",
        f2p_commands=('test "$(cat src.txt)" = fixed',),
        p2p_commands=("true",),
    )
    return repo, instance


def _run_verifier(tmp_path, repo, instance):
    import shlex
    import subprocess

    from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import _test_script

    tests = tmp_path / "injected-tests"
    tests.mkdir()
    (tests / "test.patch").write_text(instance.test_patch)
    logs = tmp_path / "logs"
    script = (
        _test_script(instance)
        .replace("/workspace/repo", shlex.quote(str(repo)))
        .replace("/logs/verifier", shlex.quote(str(logs)))
        .replace("/tests/test.patch", shlex.quote(str(tests / "test.patch")))
    )
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    return (logs / "reward.txt").read_text().strip()


def test_committed_production_fix_still_passes(tmp_path):
    repo, instance = _local_instance(tmp_path)
    (repo / "src.txt").write_text("fixed")
    _git(repo, "commit", "-qam", "fix")
    assert _run_verifier(tmp_path, repo, instance) == "1"


def test_committed_build_changes_are_rejected_even_with_replace_ref(tmp_path):
    repo, instance = _local_instance(tmp_path)
    (repo / "src.txt").write_text("fixed")
    (repo / "CMakeLists.txt").write_text("tampered")
    _git(repo, "commit", "-qam", "candidate")
    _git(repo, "replace", instance.base_commit, "HEAD")
    assert _run_verifier(tmp_path, repo, instance) == "0"


def test_missing_baseline_fails_closed(tmp_path):
    repo, instance = _local_instance(tmp_path)
    (repo / "src.txt").write_text("fixed")
    instance = replace(instance, base_commit="a" * 40)
    assert _run_verifier(tmp_path, repo, instance) == "0"


def test_clean_room_retains_only_pinned_base(tmp_path):
    import shlex
    import subprocess

    from tinker_cookbook.recipes.kokkos_rl.dataset.export_harbor import _clean_room_command

    repo, instance = _local_instance(tmp_path)
    (repo / "answer.txt").write_text("future solution")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "future solution")
    future = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", "https://example.com/unused")
    command = _clean_room_command(instance.base_commit).replace(
        "/workspace/repo", shlex.quote(str(repo))
    )
    subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
    assert _git(repo, "rev-list", "--all", "HEAD") == instance.base_commit
    assert not _git(repo, "remote")
    assert (
        subprocess.run(["git", "cat-file", "-e", future], cwd=repo, capture_output=True).returncode
        != 0
    )


def test_runtime_tasks_use_fixed_verifier_without_changing_published_export(tmp_path):
    from tinker_cookbook.recipes.kokkos_rl.rl.tasks import prepared_kokkos_tasks

    task_dir = export_instance(_instance(), tmp_path)
    published = task_dir / "tests" / "test.sh"
    published.write_text("historical verifier using HEAD")
    with prepared_kokkos_tasks(tmp_path) as tasks:
        runtime_path = tasks[0].task_dir
        assert "baseline=" in (runtime_path / "tests" / "test.sh").read_text()
        assert ".git/shallow" in (runtime_path / "environment" / "Dockerfile").read_text()
        assert published.read_text() == "historical verifier using HEAD"
    assert not runtime_path.exists()
