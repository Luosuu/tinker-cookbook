#!/bin/bash
#SBATCH --job-name=hpc-claude-wire
#SBATCH --account=lia-lab-members
#SBATCH --partition=bii-largemem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/fad3ew/openhands-smoke/claude-turn-diagnostics/wire-%j.log

source /etc/profile
set -euo pipefail
module load nodejs/24.5.0
ROOT=/scratch/fad3ew/openhands-smoke/claude-code-pass4-20260924
/scratch/fad3ew/openhands-smoke/.venv/bin/python \
  /scratch/fad3ew/openhands-smoke/claude-turn-diagnostics/diagnose_claude_wire.py \
  "$ROOT/code/claude-runtime" \
  /scratch/fad3ew/openhands-smoke/inkling-eval.env \
  "$ROOT/tasks/kokkos__kokkos-8164/instruction.md" \
  "${HPC_PROBE_EFFORT-high}"
