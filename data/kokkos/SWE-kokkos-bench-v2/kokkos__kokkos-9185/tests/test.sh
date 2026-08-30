#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only HEAD --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(containers/unit_tests/TestCreateMirror.cpp core/unit_test/CMakeLists.txt core/unit_test/TestCreateMirror.cpp core/unit_test/incremental/Test10_HierarchicalBasics.hpp core/unit_test/incremental/Test11a_ParallelFor_TeamThreadRange.hpp core/unit_test/incremental/Test11b_ParallelFor_TeamVectorRange.hpp core/unit_test/incremental/Test11c_ParallelFor_ThreadVectorRange.hpp core/unit_test/incremental/Test12a_ThreadScratch.hpp core/unit_test/incremental/Test12b_TeamScratch.hpp core/unit_test/incremental/Test13a_ParallelRed_TeamThreadRange.hpp core/unit_test/incremental/Test13b_ParallelRed_TeamVectorRange.hpp core/unit_test/incremental/Test13c_ParallelRed_ThreadVectorRange.hpp core/unit_test/incremental/Test16_ParallelScan.hpp core/unit_test/view/TestCreateMirrorViewAndCopy.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e HEAD:"$path" 2>/dev/null; then
    git checkout HEAD -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_Serial_ViewSupport --parallel; then
  exit 0
fi

f2p_commands=()
p2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial_ViewSupport --parallel')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
