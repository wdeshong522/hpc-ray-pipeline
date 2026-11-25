#!/bin/bash
# Submit the LoRA scaling grid: 3 repeats of every distinct configuration.
#   strong: 4,000 examples on 1, 2, 4, 8 GPUs
#   weak:   1,000 examples per GPU on 1, 2, 3, 4 GPUs
# A configuration that appears in both (4 GPUs, 4,000 examples) runs once.
# Usage: slurm/submit_lora_grid.sh [--dry]
set -euo pipefail

DRY=${1:-}

CONFIGS=$({
  for g in 1 2 4 8; do echo "$g 4000"; done
  for g in 1 2 3 4; do echo "$g $((1000 * g))"; done
} | sort -u -k1,1n -k2,2n)

for rep in 1 2 3; do
  while read -r g n; do
    if [ "$DRY" = "--dry" ]; then
      echo "gpus=$g examples=$n repeat=$rep"
    else
      bash slurm/submit_lora.sh "$g" "$n" "$rep"
    fi
  done <<< "$CONFIGS"
done
