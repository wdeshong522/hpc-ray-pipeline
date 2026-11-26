#!/bin/bash
# Usage: slurm/submit_generate.sh REPLICAS N_SCENARIOS REPEAT MODEL_DIR
# One replica per node, 2 GPUs (pipeline parallel size 2) per replica.
set -euo pipefail

R=$1
N=$2
REP=$3
MODEL_DIR=$4

RUN_ID="gen_r${R}_n${N}_rep${REP}"

mkdir -p results/_slurm
sbatch --nodes="$R" --ntasks="$R" \
    -J "$RUN_ID" -o "results/_slurm/%x_%j.log" \
    --export=ALL,N_SCENARIOS="$N",RUN_ID="$RUN_ID",MODEL_DIR="$MODEL_DIR" \
    slurm/generate.sbatch
