#!/usr/bin/env bash
# Run native, DSH control and DSH state-target, four draws each, all at once so every arm
# shares the same period and load. Scores drift between sessions by several points, so arms
# are only comparable within one block.
#
#   PER_RUN=3 bash notes/dsh_inkling/harness/contree/launch_pass4.sh /tmp/dsh-contree 40 04
#
# Arguments: output root, first DSH run number (eight consecutive), first native run number
# (four consecutive, zero-padded to two digits).
set -u
cd "$(dirname "$0")/../../../.."
C=${1:-/tmp/dsh-contree}
FIRST_DSH=${2:-40}
FIRST_NATIVE=${3:-04}
PER_RUN=${PER_RUN:-3}
TASKS=notes/dsh_inkling/harness/contree/tasklists/full100.txt
PERSONA=notes/dsh_inkling/harness/contree/persona-reference.txt
mkdir -p "$C/pass4-logs"

dsh=(); for i in 0 1 2 3 4 5 6 7; do dsh+=("full$((FIRST_DSH + i))"); done
native=(); for i in 0 1 2 3; do native+=("native$(printf %02d $((10#$FIRST_NATIVE + i)))"); done
for run in "${dsh[@]}" "${native[@]}"; do
  if [ -e "$C/$run" ]; then echo "refusing: $C/$run exists"; exit 1; fi
done

launch_dsh() {
  local run=$1; shift
  .venv/bin/python -m notes.dsh_inkling.harness.contree.run_batch \
    --tasks-file $TASKS --run-dir "$C/$run" --concurrency "$PER_RUN" --cpu-affinity 0,1 \
    --persona $PERSONA "$@" > "$C/pass4-logs/$run.log" 2>&1 &
}
launch_native() {
  local run=$1
  .venv/bin/python notes/dsh_inkling/harness/native/run_native_contree.py \
    data/kokkos/SWE-kokkos-bench-v2 "$C/$run" "$C/cache/images.json" "$PER_RUN" \
    > "$C/pass4-logs/$run.log" 2>&1 &
}

# Interleave arms in launch order so no arm gets a head start.
for i in 0 1 2 3; do
  launch_native "${native[$i]}"
  launch_dsh "${dsh[$i]}"
  launch_dsh "${dsh[$((i + 4))]}" --state-target
done
echo "launched 12 runs at $PER_RUN each"
wait
echo "all runs finished"
