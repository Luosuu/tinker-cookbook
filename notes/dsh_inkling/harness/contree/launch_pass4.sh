#!/usr/bin/env bash
# Run native and two DSH arms, four draws each, all at once so every arm shares the same
# period and load. Scores drift between sessions by several points, so arms are only
# comparable within one block.
#
#   PER_RUN=3 bash notes/dsh_inkling/harness/contree/launch_pass4.sh /tmp/dsh-contree 40 04
#
# Arguments: output root, first DSH run number (eight consecutive: four arm A, then four
# arm B), first native run number (four consecutive, zero-padded to two digits). Arm flags
# default to the unmodified harness (A) and --state-target (B); override with ARM_A_FLAGS and
# ARM_B_FLAGS, e.g. ARM_A_FLAGS=--state-target ARM_B_FLAGS="--state-target --managed-checkpoint".
set -u
cd "$(dirname "$0")/../../../.."
C=${1:-/tmp/dsh-contree}
FIRST_DSH=${2:-40}
FIRST_NATIVE=${3:-04}
PER_RUN=${PER_RUN:-3}
read -r -a arm_a <<< "${ARM_A_FLAGS:-}"
read -r -a arm_b <<< "${ARM_B_FLAGS:---state-target}"
TASKS=notes/dsh_inkling/harness/contree/tasklists/full100.txt
PERSONA=notes/dsh_inkling/harness/contree/persona-reference.txt
# Warm-up concurrency is deliberately low: 36 sandboxes each cloning kokkos from GitHub at once
# (which happens when the image cache is empty, e.g. after a reboot wipes /tmp) triggers clone
# failures and ConTree timeouts. Build the shared image cache first, serially-ish, so the
# parallel block below does zero fresh clones.
WARM_CONCURRENCY=${WARM_CONCURRENCY:-4}
WARM_ATTEMPTS=${WARM_ATTEMPTS:-4}
mkdir -p "$C/pass4-logs" "$C/cache"

dsh=(); for i in 0 1 2 3 4 5 6 7; do dsh+=("full$((FIRST_DSH + i))"); done
native=(); for i in 0 1 2 3; do native+=("native$(printf %02d $((10#$FIRST_NATIVE + i)))"); done
for run in "${dsh[@]}" "${native[@]}"; do
  if [ -e "$C/$run" ]; then echo "refusing: $C/$run exists"; exit 1; fi
done

# Phase 1: warm the image cache. --prepare-only builds and caches each task's image without
# running an episode; --resume --retry-errors makes repeated attempts fill in only the tasks
# that timed out, so a flaky ConTree converges instead of failing the whole block.
warm_dir="$C/warm"
for attempt in $(seq 1 "$WARM_ATTEMPTS"); do
  resume=""
  [ -e "$warm_dir" ] && resume="--resume --retry-errors"
  .venv/bin/python -m notes.dsh_inkling.harness.contree.run_batch \
    --tasks-file $TASKS --run-dir "$warm_dir" --cache "$C/cache/images.json" \
    --concurrency "$WARM_CONCURRENCY" --prepare-only $resume \
    > "$C/pass4-logs/warm-$attempt.log" 2>&1
  remaining=$(.venv/bin/python - "$warm_dir" <<'PY'
import glob, json, sys
done = {}
for p in glob.glob(sys.argv[1] + "/results/*/summary.json"):
    s = json.load(open(p))
    done[s["task_name"]] = not s.get("error")
total = sum(1 for line in open("notes/dsh_inkling/harness/contree/tasklists/full100.txt") if line.strip())
print(total - sum(1 for ok in done.values() if ok))
PY
)
  echo "warm attempt $attempt: $remaining tasks still unprepared"
  [ "$remaining" -eq 0 ] && break
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
  launch_dsh "${dsh[$i]}" ${arm_a[@]+"${arm_a[@]}"}
  launch_dsh "${dsh[$((i + 4))]}" ${arm_b[@]+"${arm_b[@]}"}
done
echo "launched 12 runs at $PER_RUN each: A=[${arm_a[*]-}] B=[${arm_b[*]-}]"
wait
echo "all runs finished"
