#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
set -euo pipefail

SIF=${SIF:-$HOME/sif/hpc.sif}
PY=/opt/venvs/serve/bin/python
export PYTHONPATH=$PWD

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
BPORT=$((30000 + SLURM_JOB_ID % 10000))
EXTRA=""
if [ "${SERIAL:-0}" = "1" ]; then EXTRA="--serial"; fi

srun --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --gpus-per-task=2 --cpus-per-task=16 \
    apptainer exec --nv "$SIF" "$PY" -m src.judge.replica \
    --run_id "$RUN_ID" --n_responses "$N" --responses "$RESPONSES" \
    --qwen_dir "$QWEN_DIR" --granite_dir "$GRANITE_DIR" \
    --head_host "$HEAD" --barrier_port "$BPORT" --concurrency "${CONCURRENCY:-64}" $EXTRA

python3 -m src.judge.join --run_dir "results/$RUN_ID"
