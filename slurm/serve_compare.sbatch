#!/bin/bash
#SBATCH -A odu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -J serve_compare

# Server A: base + adapter. Server B: merged weights. Same 2 GPUs, one after the other.
# Usage: RUN=lora_g8_n4000_rep1 sbatch slurm/serve_compare.sbatch

set -uo pipefail
SIF=${SIF:-$HOME/sif/hpc.sif}
PY=/opt/venvs/serve/bin/python
BASE=${BASE:-meta-llama/Llama-3.1-8B-Instruct}
RUN=${RUN:-lora_g8_n4000_rep1}
ADAPTER=results/$RUN/adapter
MERGED=results/merged/$RUN
PORT=$((8000 + SLURM_JOB_ID % 1000))

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/_slurm
SERVE="apptainer exec --nv $SIF bash slurm/in_serve_env.sh $PY -m vllm.entrypoints.openai.api_server"
COMMON="--pipeline-parallel-size 2 --dtype float16 --max-model-len 4096"

wait_ready() {
  for _ in $(seq 1 180); do
    curl -s "http://localhost:$1/v1/models" | grep -q "$2" && return 0
    kill -0 "$3" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

run_server() {
  local port=$1 name=$2 log=$3; shift 3
  T=$(date +%s)
  $SERVE "$@" $COMMON --port "$port" > "$log" 2>&1 &
  local pid=$!
  if wait_ready "$port" "$name" "$pid"; then
    echo "ready after $(( $(date +%s) - T )) s"
    shift 0
    PYTHONPATH="$PWD" apptainer exec "$SIF" $PY experiments/serve_compare.py \
        --port "$port" --models $MODELS --data_dir data
  else
    echo "server not ready; see $log"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}

echo "== A: base + adapter"
MODELS="dentsim $BASE" run_server "$PORT" dentsim "results/_slurm/compare_${SLURM_JOB_ID}_A.log" \
    --model "$BASE" --tokenizer "$ADAPTER" --enable-lora --lora-modules "dentsim=$ADAPTER"
sleep 20
echo "== B: merged"
MODELS="merged" run_server "$((PORT + 1))" merged "results/_slurm/compare_${SLURM_JOB_ID}_B.log" \
    --model "$MERGED" --served-model-name merged
