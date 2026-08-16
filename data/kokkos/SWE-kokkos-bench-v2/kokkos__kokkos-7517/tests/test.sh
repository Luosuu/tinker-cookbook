#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 98faff3915c87bed36bf55aac32373e2f79f42cc --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(containers/unit_tests/CMakeLists.txt containers/unit_tests/TestDynRankView_TeamScratch.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 98faff3915c87bed36bf55aac32373e2f79f42cc:"$path" 2>/dev/null; then
    git checkout 98faff3915c87bed36bf55aac32373e2f79f42cc -- "$path"
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

f2p_commands=('"$(find build -type f -name Kokkos_ContainersUnitTest_Serial -perm -111 -print -quit)" --gtest_filter='"'"'*dyn_rank_view_team_scratch*'"'"'')
p2p_commands=('"$(find build -type f -name Kokkos_ContainersUnitTest_Serial -perm -111 -print -quit)" --gtest_filter='"'"'*ContainersUnitTest_Serial*'"'"'')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
