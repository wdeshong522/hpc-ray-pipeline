#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

# Full pipeline in one allocation of K whole nodes (4K GPUs): LoRA, merge, generation, judging as srun steps.
# Every stage uses all 4K GPUs: LoRA 4K workers, generation 2K replicas, judging K units.
# Submit with slurm/submit_pipeline.sh, which sets K, N, PIPE, QWEN_DIR and GRANITE_DIR.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF=${SIF:-$HOME/sif/hpc.sif}
TRAIN_PY=/opt/venvs/train/bin/python
RAY=/opt/venvs/train/bin/ray
SERVE_PY=/opt/venvs/serve/bin/python
CPUS=16
GPUS=$((4 * K))
LORA_ID=${PIPE}_lora
GEN_ID=${PIPE}_gen
JUDGE_ID=${PIPE}_judge
MERGED=results/merged/$LORA_ID
FROZEN=results/$PIPE/judge_input.ndjson
STAGES=results/$PIPE/stages.tsv

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
HEAD_IP=$(getent ahostsv4 "$HEAD" | awk 'NR==1{print $1}')
BPORT=$((30000 + SLURM_JOB_ID % 10000))

mkdir -p "results/$PIPE" results/merged
mark() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s.%N)" >> "$STAGES"; }
fail() { mark "$1" failed; echo "stage $1 failed" >&2; exit 1; }
RAY_PIDS=()
stop_ray() { for p in "${RAY_PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; RAY_PIDS=(); }
trap stop_ray EXIT

mark job start

# Stage 1: LoRA on all 4K GPUs, through a Ray cluster started inside the allocation (4 GPUs per node).
mark lora start
export RUN_T0=$(date +%s.%N)
RPORT=$((20000 + SLURM_JOB_ID % 20000))
DASH=$((40000 + SLURM_JOB_ID % 20000))
MINW=$((10000 + (SLURM_JOB_ID % 100) * 100))
MAXW=$((MINW + 99))
srun --overlap -N1 -n1 -w "$HEAD" --gres=gpu:4 --cpus-per-task=32 \
    apptainer exec --nv "$SIF" $RAY start --head --node-ip-address="$HEAD_IP" --port=$RPORT \
    --num-cpus=32 --num-gpus=4 --dashboard-port=$DASH \
    --min-worker-port=$MINW --max-worker-port=$MAXW --disable-usage-stats --block &
RAY_PIDS+=($!)
for node in "${NODES[@]:1}"; do
    srun --overlap -N1 -n1 -w "$node" --gres=gpu:4 --cpus-per-task=32 \
        apptainer exec --nv "$SIF" $RAY start --address="$HEAD_IP:$RPORT" \
        --num-cpus=32 --num-gpus=4 --min-worker-port=$MINW --max-worker-port=$MAXW --block &
    RAY_PIDS+=($!)
done
srun --overlap -N1 -n1 -w "$HEAD" --gres=none --cpus-per-task=2 apptainer exec "$SIF" \
    $TRAIN_PY -m src.common.ray_wait --address="$HEAD_IP:$RPORT" --gpus=$GPUS --timeout=300 || fail lora
srun --overlap -N1 -n1 -w "$HEAD" --gres=none --cpus-per-task=2 apptainer exec "$SIF" \
    $TRAIN_PY -m src.train.launch --ray_address="$HEAD_IP:$RPORT" \
    --gpus=$GPUS --n_examples=$N --run_id=$LORA_ID || fail lora
stop_ray
sleep 10
mark lora end

# Merge the adapter into the base weights (CPU, D32).
mark merge start
srun -N1 -n1 -w "$HEAD" --gres=none --cpus-per-task=$CPUS --export=ALL,PYTHONPATH="$PWD" \
    apptainer exec "$SIF" $TRAIN_PY -m src.generate.merge \
    --adapter "results/$LORA_ID/adapter" --out "$MERGED" --run_id "merge_$LORA_ID" || fail merge
mark merge end

# Stage 2: 2K generation replicas, 2 GPUs each, two per node.
mark generation start
srun --kill-on-bad-exit=1 --nodes=$K --ntasks=$((2 * K)) --ntasks-per-node=2 --gpus-per-task=2 --cpus-per-task=$CPUS \
    apptainer exec --nv "$SIF" $SERVE_PY -m src.generate.replica \
    --run_id "$GEN_ID" --n_scenarios "$N" --model_dir "$MERGED" \
    --head_host "$HEAD" --barrier_port "$BPORT" --concurrency 192 || fail generation
mark generation end

mark freeze start
python3 -m src.judge.inputs --run_dir "results/$GEN_ID" --n "$N" --out "$FROZEN" || fail freeze
mark freeze end

# Stage 3: K judge units (one Qwen and one Granite replica each), 2 tasks per node.
mark judging start
srun --kill-on-bad-exit=1 --nodes=$K --ntasks=$((2 * K)) --ntasks-per-node=2 --gpus-per-task=2 --cpus-per-task=$CPUS \
    apptainer exec --nv "$SIF" $SERVE_PY -m src.judge.replica \
    --run_id "$JUDGE_ID" --n_responses "$N" --responses "$FROZEN" \
    --qwen_dir "$QWEN_DIR" --granite_dir "$GRANITE_DIR" \
    --head_host "$HEAD" --barrier_port $((BPORT + 1)) --concurrency 64 || fail judging
mark judging end

python3 -m src.judge.join --run_dir "results/$JUDGE_ID" || fail join
mark job end
