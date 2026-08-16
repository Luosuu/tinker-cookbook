#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 08f79ebe085bc6f03d222f069f9034b55e59465d --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/CMakeLists.txt core/unit_test/TestTimer.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 08f79ebe085bc6f03d222f069f9034b55e59465d:"$path" 2>/dev/null; then
    git checkout 08f79ebe085bc6f03d222f069f9034b55e59465d -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_Serial1 Kokkos_CoreUnitTest_Serial2 --parallel; then
  exit 0
fi

f2p_commands=()
p2p_commands=('./build/core/unit_test/Kokkos_CoreUnitTest_Serial1 --gtest_filter=TEST_CATEGORY.timer')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
