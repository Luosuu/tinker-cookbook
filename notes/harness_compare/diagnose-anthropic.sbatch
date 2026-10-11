#!/bin/bash
#SBATCH --job-name=hpc-claude-turns
#SBATCH --account=lia-lab-members
#SBATCH --partition=bii-largemem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/fad3ew/openhands-smoke/claude-turn-diagnostics/diagnose-%j.log

source /etc/profile
set -euo pipefail
/scratch/fad3ew/openhands-smoke/.venv/bin/python \
  /scratch/fad3ew/openhands-smoke/claude-turn-diagnostics/diagnose_anthropic_turns.py \
  /scratch/fad3ew/openhands-smoke/inkling-eval.env
