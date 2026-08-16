#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 765cd9d1fa9a01d3cb6ba3b0d2a4d4914d5060d1 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestViewCtorProp.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 765cd9d1fa9a01d3cb6ba3b0d2a4d4914d5060d1:"$path" 2>/dev/null; then
    git checkout 765cd9d1fa9a01d3cb6ba3b0d2a4d4914d5060d1 -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_Serial2 --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial2')
p2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial2' './build/core/unit_test/Kokkos_CoreUnitTest_Serial2 --gtest_filter='"'"'TEST(TEST_CATEGORY, vcp_pointer_add_property)'"'"'')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
