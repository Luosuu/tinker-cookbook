#!/bin/bash
#SBATCH --partition=bii-largemem
#SBATCH --account=lia-lab-members
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=dsh-inkling
#SBATCH --output=/scratch/fad3ew/openhands-smoke/dsh-iteration/logs/%x-%A_%a.log

# One episode per array element, matching the reference 4-cpu/16GB/2h envelope.
# Usage:  sbatch --array=0-N --export=ALL,DSH_ARRAY=<taskfile> smoke.sbatch
source /etc/profile
set -euo pipefail
module load apptainer/1.4.5 nodejs/24.5.0

BASE=/scratch/fad3ew/openhands-smoke
I=$BASE/dsh-iteration
export APPTAINER_CACHEDIR=$BASE/cache APPTAINER_TMPDIR=$BASE/tmp
export PYTHONUNBUFFERED=1 OPENHANDS_SUPPRESS_BANNER=1

TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${DSH_TASKS:?set DSH_TASKS to a task list file}")
RUN="${DSH_RUN:?set DSH_RUN to the run directory}"
OUT="$RUN/results/$TASK"
mkdir -p "$OUT"

exec "$BASE/.venv/bin/python" "$I/driver/run_task.py" "$TASK" "$OUT" \
  --python "$BASE/.venv/bin/python" \
  --tasks-root "$I/tasks" \
  --code-root "$I/code" \
  --runtime "$I/dsh-runtime" \
  --proxy-script "$I/proxy/proxy.py" \
  --env-file "$BASE/inkling-eval.env" \
  --upstream "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1" \
  --profile-patch "$I/dsh-runtime/dsh.patch.yml.in" \
  --persona-file "${DSH_PERSONA_FILE:-$I/dsh-runtime/persona-reference.txt}" \
  --node node \
  --max-turns "${DSH_MAX_TURNS:-40}" \
  --max-tool-calls "${DSH_MAX_TOOL_CALLS:-80}" \
  --max-tokens "${DSH_MAX_TOKENS:-65536}"
