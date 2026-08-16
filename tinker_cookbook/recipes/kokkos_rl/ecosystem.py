"""Repository profiles for Kokkos ecosystem dataset construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepositoryProfile:
    repo: str
    default_branch: str
    source_prefixes: tuple[str, ...]
    test_dir_names: frozenset[str]
    configure_command: str
    build_system: str = "cmake"
    full_build_for_validation: bool = False


_KOKKOS_CONFIGURE = (
    "cmake -S . -B build -G Ninja "
    "-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON "
    "-DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF "
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
)

REPOSITORY_PROFILES = {
    profile.repo: profile
    for profile in (
        RepositoryProfile(
            repo="kokkos/kokkos",
            default_branch="develop",
            source_prefixes=("core/src/", "containers/src/", "algorithms/src/"),
            test_dir_names=frozenset({"unit_test", "unit_tests"}),
            configure_command=_KOKKOS_CONFIGURE,
        ),
        RepositoryProfile(
            repo="kokkos/kokkos-kernels",
            default_branch="develop",
            source_prefixes=(
                "batched/",
                "blas/",
                "common/",
                "graph/",
                "lapack/",
                "ode/",
                "sparse/",
            ),
            test_dir_names=frozenset({"unit_test", "unit_tests"}),
            configure_command=(
                "cmake -S . -B build -G Ninja -DKokkosKernels_ENABLE_TESTS=ON "
                "-DKokkos_ROOT=/opt/kokkos -DCMAKE_BUILD_TYPE=RelWithDebInfo "
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
            ),
        ),
        RepositoryProfile(
            repo="kokkos/kokkos-tools",
            default_branch="develop",
            source_prefixes=("common/", "debugging/", "profiling/"),
            test_dir_names=frozenset({"test", "tests", "unit_test", "unit_tests"}),
            configure_command=(
                "cmake -S . -B build -G Ninja -DKokkos_ROOT=/opt/kokkos "
                "-DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo"
            ),
            full_build_for_validation=True,
        ),
        RepositoryProfile(
            repo="kokkos/pykokkos",
            default_branch="main",
            source_prefixes=("base/include/", "base/src/", "pykokkos/"),
            test_dir_names=frozenset({"test", "tests"}),
            configure_command=("python -m pip install --break-system-packages -e . --no-deps"),
            build_system="python",
        ),
        RepositoryProfile(
            repo="kokkos/kokkos-remote-spaces",
            default_branch="main",
            source_prefixes=("src/",),
            test_dir_names=frozenset({"unit_test", "unit_tests"}),
            configure_command=(
                "cmake -S . -B build -G Ninja -DKokkos_ROOT=/opt/kokkos "
                "-DKRS_ENABLE_MPISPACE=ON -DKRS_ENABLE_TESTS=ON "
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            ),
            full_build_for_validation=True,
        ),
        RepositoryProfile(
            repo="kokkos/kokkos-resilience",
            default_branch="main",
            source_prefixes=("src/",),
            test_dir_names=frozenset({"test", "tests", "unit_test", "unit_tests"}),
            configure_command=(
                "cmake -S . -B build -G Ninja -DKokkos_ROOT=/opt/kokkos "
                "-DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo"
            ),
            full_build_for_validation=True,
        ),
    )
}


def get_repository_profile(repo: str) -> RepositoryProfile:
    try:
        return REPOSITORY_PROFILES[repo]
    except KeyError:
        supported = ", ".join(sorted(REPOSITORY_PROFILES))
        raise ValueError(f"unsupported repository {repo!r}; choose one of: {supported}") from None


def instance_id_for(repo: str, pr_number: int) -> str:
    return f"{repo.replace('/', '__')}-{pr_number}"
