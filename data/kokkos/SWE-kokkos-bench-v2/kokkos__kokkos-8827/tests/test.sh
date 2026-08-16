#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only a5d374ba179921b9805008dda06a4b690cfbdd1a --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestMathematicalFunctions.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e a5d374ba179921b9805008dda06a4b690cfbdd1a:"$path" 2>/dev/null; then
    git checkout a5d374ba179921b9805008dda06a4b690cfbdd1a -- "$path"
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

f2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial1')
p2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial1 && ctest --test-dir build -R mathematical_functions_binary_predicates --output-on-failure')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
