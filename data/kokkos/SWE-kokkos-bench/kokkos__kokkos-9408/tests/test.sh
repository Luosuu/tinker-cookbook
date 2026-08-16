#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only dcc7879ea2ecadea6b6cdca0c5f815e0ecd0bd77 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/view/TestAccessorFromMemoryTraits.cpp core/unit_test/view/TestBasicViewMDSpanConversion.cpp core/unit_test/view/TestReferenceCountedAccessor.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e dcc7879ea2ecadea6b6cdca0c5f815e0ecd0bd77:"$path" 2>/dev/null; then
    git checkout dcc7879ea2ecadea6b6cdca0c5f815e0ecd0bd77 -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! cmake --build build --target Kokkos_CoreTestCompileOnly Kokkos_CoreUnitTest_Serial_ViewSupport --parallel; then
  exit 0
fi

f2p_commands=()
p2p_commands=('ctest --test-dir build -R '"'"'^Kokkos_CoreUnitTest_Serial_ViewSupport$'"'"' --output-on-failure')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
