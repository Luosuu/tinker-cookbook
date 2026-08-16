#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 0d40eccc4c79d463279045fca1624873a682795e --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestSubView_a.hpp core/unit_test/TestViewSubview.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 0d40eccc4c79d463279045fca1624873a682795e:"$path" 2>/dev/null; then
    git checkout 0d40eccc4c79d463279045fca1624873a682795e -- "$path"
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

f2p_commands=('"$(find build -type f -name Kokkos_CoreUnitTest_Serial2 -perm -111 -print -quit)" --gtest_filter='"'"'*view_subview_wrong_extents*'"'"'')
p2p_commands=('"$(find build -type f -name Kokkos_CoreUnitTest_Serial2 -perm -111 -print -quit)" --gtest_filter='"'"'*view_static_tests*'"'"'')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
