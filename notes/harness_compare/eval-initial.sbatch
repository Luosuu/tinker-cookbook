#!/bin/bash
#SBATCH --job-name=hpc-dsh-pilot
#SBATCH --account=lia-lab-members
#SBATCH --partition=bii-largemem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/fad3ew/openhands-smoke/harness-cross-eval-20260924/slurm-%A_%a.log

source /etc/profile
set -euo pipefail
module load apptainer/1.4.5 nodejs/24.5.0

BASE=/scratch/fad3ew/openhands-smoke/harness-cross-eval-20260924
PI_ROOT=${HPC_PI_EVAL_ROOT:-/scratch/fad3ew/openhands-smoke/pi-skill-eval-20260922}
TASKS=(kokkos__kokkos-8594 kokkos__kokkos-9276)
export HPC_PI_EVAL_ROOT="$PI_ROOT"
export PYTHONPATH="$PI_ROOT/code"
export PYTHONUNBUFFERED=1

/scratch/fad3ew/openhands-smoke/.venv/bin/python \
  "$BASE/eval_deepseek.py" \
  "${TASKS[$SLURM_ARRAY_TASK_ID]}" \
  "$BASE/results/${SLURM_ARRAY_TASK_ID}"
