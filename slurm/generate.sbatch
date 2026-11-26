#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
set -euo pipefail

SIF=${SIF:-$HOME/sif/hpc.sif}
PY=/opt/venvs/serve/bin/python
export PYTHONPATH=$PWD

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
BPORT=$((30000 + SLURM_JOB_ID % 10000))

srun --kill-on-bad-exit=1 --ntasks-per-node=1 --gres=gpu:2 \
    apptainer exec --nv "$SIF" "$PY" -m src.generate.replica \
    --run_id "$RUN_ID" --n_scenarios "$N_SCENARIOS" --model_dir "$MODEL_DIR" \
    --head_host "$HEAD" --barrier_port "$BPORT" --concurrency "${CONCURRENCY:-12}"
