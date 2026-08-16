#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only b33c89a21247ec0011a219571114cb33fbd1de55 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(containers/unit_tests/CMakeLists.txt containers/unit_tests/TestDynRankView_ViewCustomization.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e b33c89a21247ec0011a219571114cb33fbd1de55:"$path" 2>/dev/null; then
    git checkout b33c89a21247ec0011a219571114cb33fbd1de55 -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_ContainersUnitTest_Serial --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_ContainersUnitTest_Serial --parallel')
p2p_commands=('cmake --build build --target Kokkos_ContainersUnitTest_Serial --parallel')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
