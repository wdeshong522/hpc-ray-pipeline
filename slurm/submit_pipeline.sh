#!/bin/bash
# Usage: slurm/submit_pipeline.sh K N REPEAT
# One allocation of K whole nodes (4K GPUs); weak scaling uses N = 1000 * K, strong scaling N = 4000.
# Set DEP=<jobid> to start only after that job succeeds.
set -euo pipefail

K=$1
N=$2
REP=$3
QWEN_DIR=${QWEN_DIR:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
GRANITE_DIR=${GRANITE_DIR:-$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.1-8b/snapshots/1504002f650e656a0a3789d99574df12e3e94ed0}
PIPE="pipe_k${K}_n${N}_rep${REP}"

if [ -e "results/$PIPE" ] || [ -e "results/${PIPE}_lora" ]; then
    echo "results/$PIPE already exists; move it away first" >&2
    exit 1
fi

DEP_ARG=""
if [ -n "${DEP:-}" ]; then DEP_ARG="--dependency=afterok:$DEP"; fi

mkdir -p results/_slurm
sbatch --parsable --nodes="$K" -t 05:00:00 $DEP_ARG -J "$PIPE" -o "results/_slurm/%x_%j.log" \
    --export=ALL,K="$K",N="$N",PIPE="$PIPE",QWEN_DIR="$QWEN_DIR",GRANITE_DIR="$GRANITE_DIR" \
    slurm/pipeline.sbatch
