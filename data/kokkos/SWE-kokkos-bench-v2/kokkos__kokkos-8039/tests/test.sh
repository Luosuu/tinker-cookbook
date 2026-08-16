#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only c0a3d7e4a417c02bd4c45bd6f8b9bdd51499f014 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(algorithms/unit_tests/TestRandom.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e c0a3d7e4a417c02bd4c45bd6f8b9bdd51499f014:"$path" 2>/dev/null; then
    git checkout c0a3d7e4a417c02bd4c45bd6f8b9bdd51499f014 -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_UnitTest_Random --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_UnitTest_Random')
p2p_commands=('cmake --build build --target Kokkos_UnitTest_Random && ./build/algorithms/unit_tests/Kokkos_UnitTest_Random --gtest_filter='"'"'TEST(TEST_CATEGORY,Multi_streams):TEST(TEST_CATEGORY,Random_XorShift64)'"'"'')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
