#!/bin/bash
# Usage: slurm/submit_lora.sh GPUS N_EXAMPLES REPEAT
# GPUs per node is capped at 4; more than 4 GPUs means more nodes.
set -euo pipefail

G=$1
N=$2
REP=$3

GPN=$(( G < 4 ? G : 4 ))
NODES=$(( (G + 3) / 4 ))
RUN_ID="lora_g${G}_n${N}_rep${REP}"

mkdir -p results/_slurm
sbatch -N "$NODES" --gres=gpu:"$GPN" -c $((8 * GPN)) \
    -J "$RUN_ID" -o "results/_slurm/%x_%j.log" \
    --export=ALL,GPUS="$G",N_EXAMPLES="$N",RUN_ID="$RUN_ID" \
    slurm/lora.sbatch
