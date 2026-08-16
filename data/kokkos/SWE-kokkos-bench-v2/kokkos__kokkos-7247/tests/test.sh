#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 1f99b1d3851c780aa18829ff4bf2b9b61bb9df21 --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(core/unit_test/TestTeam.hpp core/unit_test/TestTeamBasic.hpp core/unit_test/TestViewAPI.hpp core/unit_test/TestViewMapping_a.hpp)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 1f99b1d3851c780aa18829ff4bf2b9b61bb9df21:"$path" 2>/dev/null; then
    git checkout 1f99b1d3851c780aa18829ff4bf2b9b61bb9df21 -- "$path"
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

f2p_commands=('grep -q '"'"'std::remove_reference_t'"'"' core/src/impl/Kokkos_Tools_Generic.hpp')
p2p_commands=()
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
