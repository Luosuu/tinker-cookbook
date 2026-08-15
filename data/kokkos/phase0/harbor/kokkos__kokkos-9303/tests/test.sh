#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(git diff --name-only 7520672d9b29d727abb9b1a368267e576c9fe713 --)
if printf '%s
' "$illegal" | grep -E '(^|/)(unit_tests?|CMakeLists\.txt)(/|$)|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

for path in core/unit_test containers/unit_tests algorithms/unit_tests; do
  if git cat-file -e 7520672d9b29d727abb9b1a368267e576c9fe713:"$path" 2>/dev/null; then
    git checkout 7520672d9b29d727abb9b1a368267e576c9fe713 -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_Serial_ViewSupport --parallel; then
  exit 0
fi

f2p_commands=('build/core/unit_test/Kokkos_CoreUnitTest_Serial_ViewSupport --gtest_filter=serial.view_mdspan_args_subview')
p2p_commands=('ctest --test-dir build -R '"'"'^Kokkos_CoreUnitTest_Serial_ViewSupport$'"'"' --output-on-failure')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
