#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/repo
reward=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward"
cd "$repo"

illegal=$(
  git diff --name-only 933cbcb26b6b56bf1c027ce137afc1c54ba73dcd --
  git ls-files --others --exclude-standard | sed '\#^build/#d'
)
if printf '%s
' "$illegal" | grep -E '(^|/)(tests?|unit_tests?)(/|$)|(^|/)(test_.*|.*_test\.py)$|(^|/)CMakeLists\.txt$|^\.github/|^cmake/'; then
  echo "candidate patch changes protected test/build/CI files" >&2
  exit 0
fi

protected_paths=(tests/test_typeinference.py)
for path in "${protected_paths[@]}"; do
  if git cat-file -e 933cbcb26b6b56bf1c027ce137afc1c54ba73dcd:"$path" 2>/dev/null; then
    git checkout 933cbcb26b6b56bf1c027ce137afc1c54ba73dcd -- "$path"
  else
    rm -f -- "$path"
  fi
done

if ! git apply --whitespace=nowarn /tests/test.patch; then
  echo "failed to inject hidden tests" >&2
  exit 0
fi
if ! python -m compileall -q pykokkos/interface/views.py tests/test_typeinference.py; then
  exit 0
fi

f2p_commands=('python -m pytest -q tests/test_typeinference.py -k test_numpy_noncontiguous')
p2p_commands=('python -m compileall -q pykokkos/interface/views.py tests/test_typeinference.py')
for command in "${f2p_commands[@]}" "${p2p_commands[@]}"; do
  if [[ -n "$command" ]] && ! bash -lc "$command"; then
    exit 0
  fi
done

echo 1 > "$reward"
