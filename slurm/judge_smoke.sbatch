#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 00:50:00
set -euo pipefail

SIF=${SIF:-$HOME/sif/hpc.sif}
PY=/opt/venvs/serve/bin/python
export PYTHONPATH=$PWD

mkdir -p results/_slurm
srun --ntasks=1 --gres=gpu:2 apptainer exec --nv "$SIF" "$PY" -m src.judge.smoke \
    --model_dir "$MODEL_DIR" --served_name "$TAG" --tag "$TAG" \
    --responses "$RESPONSES" --n "${N:-60}" --concurrencies "${CONCURRENCIES:-12,32,64,128}"
