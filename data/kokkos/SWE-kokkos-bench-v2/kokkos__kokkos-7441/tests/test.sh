#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only HEAD --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestCommonPolicyInterface.hpp core/unit_test/TestMDRangePolicyCTAD.cpp core/unit_test/TestRange.hpp core/unit_test/TestRangePolicyCTAD.cpp core/unit_test/TestRangePolicyConstructors.hpp core/unit_test/TestSharedAlloc.hpp core/unit_test/TestTeamPolicyCTAD.cpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e HEAD:"$path" 2>/dev/null; then
    git checkout HEAD -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreTestCompileOnly --parallel; then
  exit 0
fi

f2p_commands=('cmake --build build --target Kokkos_CoreTestCompileOnly')
p2p_commands=('cmake --build build --target Kokkos_CoreTestCompileOnly')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
