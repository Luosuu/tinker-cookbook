#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(git diff --name-only 2f6aa91b7628d3d52f247f2540b1164c9adf0db6 --)
if printf '%s
' "$illegal" | grep -E '(^|/)(unit_tests?|CMakeLists\.txt)(/|$)|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

for path in core/unit_test containers/unit_tests algorithms/unit_tests; do
  if git cat-file -e 2f6aa91b7628d3d52f247f2540b1164c9adf0db6:"$path" 2>/dev/null; then
    git checkout 2f6aa91b7628d3d52f247f2540b1164c9adf0db6 -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_InitializeFinalize --parallel; then
  exit 0
fi

f2p_commands=()
p2p_commands=()
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
