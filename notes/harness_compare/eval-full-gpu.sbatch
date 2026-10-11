#!/bin/bash
#SBATCH --job-name=claude-pass4-gpu
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%1
#SBATCH --output=/scratch/fad3ew/openhands-smoke/claude-code-pass4-20260924/slurm/gpu-%A_%a.log

source /etc/profile
set -euo pipefail
module load apptainer/1.4.5 nodejs/24.5.0
BASE=/scratch/fad3ew/openhands-smoke
RUN="${HPC_EVAL_RUN_ROOT:-$BASE/claude-code-pass4-20260924}"
RESULTS=${HPC_RESULT_ROOT:-$RUN/results}
export HPC_PI_EVAL_ROOT="$RUN"
export PYTHONPATH="$BASE/pi-skill-eval-20260922/code"
export PYTHONUNBUFFERED=1
export APPTAINER_CACHEDIR="$BASE/cache"
export APPTAINER_TMPDIR="$BASE/tmp"

read -r TASK SAMPLE < <("$BASE/.venv/bin/python" - "$RUN/gpu-index.json" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))[int(sys.argv[2])]
print(row['task'],row['sample_index'])
PY
)
"$BASE/.venv/bin/python" "$RUN/code/eval_claude.py" \
  "$TASK" "$RESULTS/$TASK/sample-$SAMPLE" "$SAMPLE"
if [[ "${HPC_SMOKE_STRICT:-0}" == "1" ]]; then
  "$BASE/.venv/bin/python" "${HPC_CODE_ROOT:-$BASE/harness-cross-eval-20260924}/verify_smoke.py" \
    "$RESULTS/$TASK/sample-$SAMPLE"
fi
