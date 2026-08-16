#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only f58d70314ce28b4ccff5b078e2187226e48f09b2 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/CMakeLists.txt core/unit_test/TestSubView_c15.hpp core/unit_test/TestSubView_c16.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e f58d70314ce28b4ccff5b078e2187226e48f09b2:"$path" 2>/dev/null; then
    git checkout f58d70314ce28b4ccff5b078e2187226e48f09b2 -- "$path"
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

f2p_commands=()
p2p_commands=('./build/core/unit_test/Kokkos_CoreUnitTest_Serial2 --gtest_filter=SerialDeath.view_subview_constructor_layout_compatibility')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
