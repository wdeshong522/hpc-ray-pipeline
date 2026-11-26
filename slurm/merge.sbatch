#!/bin/bash
#SBATCH -A odu
#SBATCH -p main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J merge

# Usage: ADAPTER=results/lora_g8_n4000_rep1/adapter OUT=results/merged/lora_g8_n4000_rep1 sbatch slurm/merge.sbatch

set -uo pipefail
SIF=${SIF:-$HOME/sif/hpc.sif}
ADAPTER=${ADAPTER:?set ADAPTER}
OUT=${OUT:?set OUT}

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/_slurm

PYTHONPATH="$PWD" apptainer exec "$SIF" /opt/venvs/train/bin/python -m src.generate.merge \
    --adapter "$ADAPTER" --out "$OUT" --run_id "merge_$(basename "$OUT")"
