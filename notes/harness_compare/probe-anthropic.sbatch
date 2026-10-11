#!/bin/bash
#SBATCH --job-name=hpc-anthropic-probe
#SBATCH --account=lia-lab-members
#SBATCH --partition=bii-largemem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/fad3ew/openhands-smoke/harness-cross-eval-20260924/anthropic-probe-%j.log

source /etc/profile
set -euo pipefail
export TINKER_ENV_FILE=/scratch/fad3ew/openhands-smoke/inkling-eval.env
/scratch/fad3ew/openhands-smoke/.venv/bin/python \
  /scratch/fad3ew/openhands-smoke/harness-cross-eval-20260924/probe_anthropic.py
