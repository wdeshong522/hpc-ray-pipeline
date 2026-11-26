#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00

# Stage 1: start a Ray cluster inside the container, then run the LoRA driver.
# Submit with slurm/submit_lora.sh, which sets nodes, GPUs and the variables
# GPUS, N_EXAMPLES and RUN_ID.

set -uo pipefail
export RUN_T0=$(date +%s.%N)

SIF=${SIF:-$HOME/sif/hpc.sif}
RAY=/opt/venvs/train/bin/ray
PY=/opt/venvs/train/bin/python
CTR="apptainer exec --nv $SIF"

cd "$SLURM_SUBMIT_DIR"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR==1{print $1}')
PORT=$((20000 + SLURM_JOB_ID % 20000))
DASH=$((40000 + SLURM_JOB_ID % 20000))
MINW=$((10000 + (SLURM_JOB_ID % 100) * 100))
MAXW=$((MINW + 99))
GPN=$(( GPUS < 4 ? GPUS : 4 ))

trap 'kill $(jobs -p) 2>/dev/null' EXIT

srun --overlap -N1 -n1 -w "$HEAD" --gres=gpu:$GPN $CTR $RAY start --head \
    --node-ip-address="$HEAD_IP" --port=$PORT \
    --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=$GPN \
    --dashboard-port=$DASH --min-worker-port=$MINW --max-worker-port=$MAXW \
    --disable-usage-stats --block &

for node in "${NODES[@]:1}"; do
  srun --overlap -N1 -n1 -w "$node" --gres=gpu:$GPN $CTR $RAY start \
      --address="$HEAD_IP:$PORT" \
      --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=$GPN \
      --min-worker-port=$MINW --max-worker-port=$MAXW --block &
done

srun --overlap -N1 -n1 -w "$HEAD" --gres=none apptainer exec "$SIF" \
    $PY -m src.common.ray_wait --address="$HEAD_IP:$PORT" \
    --gpus=$GPUS --timeout=300 || exit 1

srun --overlap -N1 -n1 -w "$HEAD" --gres=none apptainer exec "$SIF" \
    $PY -m src.train.launch --ray_address="$HEAD_IP:$PORT" \
    --gpus=$GPUS --n_examples=$N_EXAMPLES --run_id=$RUN_ID
exit $?
