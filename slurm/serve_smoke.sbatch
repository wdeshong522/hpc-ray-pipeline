#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 00:45:00
#SBATCH -J serve_smoke

# One Llama 3.1 8B replica, pipeline parallel over 2 GPUs, with a Stage 1 adapter.
# Usage: ADAPTER=results/lora_g8_n4000_rep1/adapter sbatch slurm/serve_smoke.sbatch

set -uo pipefail
T0=$(date +%s)

SIF=${SIF:-$HOME/sif/hpc.sif}
PY=/opt/venvs/serve/bin/python
BASE=${BASE:-meta-llama/Llama-3.1-8B-Instruct}
ADAPTER=${ADAPTER:-results/lora_g8_n4000_rep1/adapter}
NAME=dentsim
PORT=$((8000 + SLURM_JOB_ID % 1000))
LOG=results/_slurm/serve_${SLURM_JOB_ID}_server.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/_slurm

trap 'kill $(jobs -p) 2>/dev/null' EXIT

apptainer exec --nv "$SIF" bash slurm/in_serve_env.sh $PY -m vllm.entrypoints.openai.api_server \
    --model "$BASE" --tokenizer "$ADAPTER" \
    --enable-lora --lora-modules "$NAME=$ADAPTER" \
    --pipeline-parallel-size 2 --dtype float16 --max-model-len 4096 \
    --port "$PORT" > "$LOG" 2>&1 &
SERVER=$!

READY=0
for _ in $(seq 1 180); do
  if curl -s "http://localhost:$PORT/v1/models" | grep -q "$NAME"; then READY=1; break; fi
  kill -0 "$SERVER" 2>/dev/null || break
  sleep 5
done
if [ "$READY" -ne 1 ]; then echo "server not ready; see $LOG"; exit 1; fi
echo "server ready after $(( $(date +%s) - T0 )) s"

PYTHONPATH="$PWD" apptainer exec "$SIF" $PY experiments/serve_smoke.py \
    --port "$PORT" --adapter_name "$NAME" --base_name "$BASE" --data_dir data
echo "total $(( $(date +%s) - T0 )) s"
