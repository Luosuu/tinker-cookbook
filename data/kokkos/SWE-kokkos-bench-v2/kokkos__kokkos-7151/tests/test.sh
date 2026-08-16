#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 277c616e3309dbd5637e0a2282fe41b6f6eb4445 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestRangePolicyConstructors.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 277c616e3309dbd5637e0a2282fe41b6f6eb4445:"$path" 2>/dev/null; then
    git checkout 277c616e3309dbd5637e0a2282fe41b6f6eb4445 -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreUnitTest_Serial1 --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial1')
p2p_commands=('cmake --build build --target Kokkos_CoreUnitTest_Serial1')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
