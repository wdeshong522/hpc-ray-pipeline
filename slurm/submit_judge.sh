#!/bin/bash
# Usage: slurm/submit_judge.sh MODE N REPEAT     MODE = serial | u1 | u2 | u3
# serial = 2 GPUs, both judges in turn; uN = N units of 1 Qwen + 1 Granite replica (4N GPUs).
# Prints the job id. Set DEP=<jobid> to start only after that job succeeds.
set -euo pipefail

MODE=$1
N=$2
REP=$3
QWEN_DIR=${QWEN_DIR:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
GRANITE_DIR=${GRANITE_DIR:-$HOME/.cache/huggingface/hub/models--ibm-granite--granite-4.1-8b/snapshots/1504002f650e656a0a3789d99574df12e3e94ed0}
RESPONSES=${RESPONSES:-results/judge_input/responses_gen_r1_n4000_c192_rep1.ndjson}

case "$MODE" in
    serial) NTASKS=1; SERIAL=1; LIMIT=05:00:00 ;;
    u1) NTASKS=2; SERIAL=0; LIMIT=03:30:00 ;;
    u2) NTASKS=4; SERIAL=0; LIMIT=02:30:00 ;;
    u3) NTASKS=6; SERIAL=0; LIMIT=02:00:00 ;;
    *) echo "MODE must be serial, u1, u2 or u3" >&2; exit 1 ;;
esac
if [ "$N" -lt 1000 ]; then LIMIT=00:40:00; fi

RUN_ID="judge_${MODE}_n${N}_rep${REP}"
if [ -e "results/$RUN_ID" ]; then
    echo "results/$RUN_ID already exists; move it away first" >&2
    exit 1
fi

DEP_ARG=""
if [ -n "${DEP:-}" ]; then DEP_ARG="--dependency=afterok:$DEP"; fi

mkdir -p results/_slurm
sbatch --parsable --ntasks="$NTASKS" -t "$LIMIT" $DEP_ARG \
    -J "$RUN_ID" -o "results/_slurm/%x_%j.log" \
    --export=ALL,N="$N",RUN_ID="$RUN_ID",SERIAL="$SERIAL",QWEN_DIR="$QWEN_DIR",GRANITE_DIR="$GRANITE_DIR",RESPONSES="$RESPONSES",CONCURRENCY=64 \
    slurm/judge.sbatch
