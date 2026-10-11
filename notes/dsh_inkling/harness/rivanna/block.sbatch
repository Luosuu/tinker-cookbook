#!/bin/bash
#SBATCH --account=lia-lab-members
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --job-name=dsh-block
# One block slot per array element. Partition, GRES, array range and log path are set by
# launch_block.sh. Requires DSH_BLOCK and DSH_POOL (cpu or gpu).
source /etc/profile
set -euo pipefail
module load apptainer/1.4.5 nodejs/24.5.0
BASE=/scratch/fad3ew/openhands-smoke
export APPTAINER_CACHEDIR=$BASE/cache APPTAINER_TMPDIR=$BASE/tmp
export PYTHONUNBUFFERED=1 OPENHANDS_SUPPRESS_BANNER=1
BLOCK="${DSH_BLOCK:?set DSH_BLOCK}"
cd "$BLOCK/code"
exec "$BASE/.venv/bin/python" -m notes.dsh_inkling.harness.rivanna.block run "$BLOCK" \
  --pool "${DSH_POOL:?set DSH_POOL}"
