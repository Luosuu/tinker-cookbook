#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only f26de9909fdc2539e36bee54ce57638c3a1eb92a --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(algorithms/unit_tests/TestRandomAccessIterator.cpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e f26de9909fdc2539e36bee54ce57638c3a1eb92a:"$path" 2>/dev/null; then
    git checkout f26de9909fdc2539e36bee54ce57638c3a1eb92a -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_AlgorithmsUnitTest_StdSet_A --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_AlgorithmsUnitTest_StdSet_A')
p2p_commands=('ctest --test-dir build -R Kokkos_AlgorithmsUnitTest_StdSet_A --output-on-failure' 'build/algorithms/unit_tests/Kokkos_AlgorithmsUnitTest_StdSet_A --gtest_filter=random_access_iterator_test.*')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
