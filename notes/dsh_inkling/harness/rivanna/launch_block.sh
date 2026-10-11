#!/usr/bin/env bash
# Submit a planned block: SIF freeze first, then the CPU and GPU episode arrays after it.
#
#   bash notes/dsh_inkling/harness/rivanna/launch_block.sh BLOCK [CPU_CONCURRENCY] [GPU_CONCURRENCY]
#
# Concurrency is fixed for the whole block; it is a controlled variable, so do not change it
# between blocks that will be combined.
set -euo pipefail
export PATH=/opt/slurm/25.05.6_mam/bin:$PATH
BLOCK=$(cd "$1" && pwd)
CPU_CONC=${2:-36}
GPU_CONC=${3:-2}
HERE="$BLOCK/code/notes/dsh_inkling/harness/rivanna"
mkdir -p "$BLOCK/logs"
count() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1"; }
NCPU=$(count "$BLOCK/slots-cpu.json")
NGPU=$(count "$BLOCK/slots-gpu.json")

freeze=$(sbatch --parsable --partition=standard --account=lia-lab-members \
  --cpus-per-task=2 --mem=4G --time=01:00:00 --job-name=dsh-freeze \
  --output="$BLOCK/logs/freeze-%j.log" \
  --wrap="cd $BLOCK/code && /scratch/fad3ew/openhands-smoke/.venv/bin/python -m notes.dsh_inkling.harness.rivanna.block freeze $BLOCK")
jobs="{\"freeze\": \"$freeze\""
if [ "$NCPU" -gt 0 ]; then
  cpu=$(sbatch --parsable --dependency=afterok:$freeze --partition=bii-largemem \
    --array=0-$((NCPU - 1))%$CPU_CONC --output="$BLOCK/logs/cpu-%A_%a.log" \
    --export=ALL,DSH_BLOCK=$BLOCK,DSH_POOL=cpu "$HERE/block.sbatch")
  jobs="$jobs, \"cpu\": \"$cpu\""
fi
if [ "$NGPU" -gt 0 ]; then
  gpu=$(sbatch --parsable --dependency=afterok:$freeze --partition=gpu-mig \
    --gres=gpu:1g.24gb:1 --array=0-$((NGPU - 1))%$GPU_CONC \
    --output="$BLOCK/logs/gpu-%A_%a.log" \
    --export=ALL,DSH_BLOCK=$BLOCK,DSH_POOL=gpu "$HERE/block.sbatch")
  jobs="$jobs, \"gpu\": \"$gpu\""
fi
jobs="$jobs, \"cpu_concurrency\": $CPU_CONC, \"gpu_concurrency\": $GPU_CONC}"
echo "$jobs" > "$BLOCK/submission.json"
echo "$jobs"
