#!/bin/bash
#SBATCH --job-name=claude-pass4-freeze
#SBATCH --account=lia-lab-members
#SBATCH --partition=bii-largemem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/fad3ew/openhands-smoke/harness-cross-eval-20260924/full-freeze-%j.log

source /etc/profile
set -euo pipefail
BASE=/scratch/fad3ew/openhands-smoke
CODE="${HPC_CODE_ROOT:-$BASE/harness-cross-eval-20260924}"
export PYTHONPATH="$BASE/pi-skill-eval-20260922/code"
export PYTHONUNBUFFERED=1

"$BASE/.venv/bin/python" \
  "$CODE/freeze_full.py" \
  "$BASE/inkling-small-round02" \
  "${HPC_FREEZE_DESTINATION:-$BASE/claude-code-pass4-20260924}" \
  "$CODE" \
  "$CODE/claude-runtime" \
  "${HPC_CONTINUATION_POLICY:-none}" \
  "${HPC_MAX_CONTINUATIONS:-0}"
