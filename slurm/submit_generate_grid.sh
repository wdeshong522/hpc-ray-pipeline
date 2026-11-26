#!/bin/bash
# Submit the generation scaling grid, one repeat per call, at 192 requests in flight (D35, D36).
#   strong: 4,000 scenarios on 1, 2, 3, 4, 6 replicas
#   weak:   1,000 scenarios per replica on 1, 2, 3, 4 replicas (4 replicas, 4,000 is the strong point)
# Order packs the 14-GPU cap: 12+2, then 8+6, then 4 with the weak runs.
# Usage: slurm/submit_generate_grid.sh REPEAT MODEL_DIR [--dry]
set -uo pipefail

REP=$1
MODEL_DIR=$2
DRY=${3:-}
C=192

CONFIGS="6 4000
1 4000
4 4000
3 4000
2 4000
3 3000
2 2000
1 1000"

while read -r r n; do
    if [ "$DRY" = "--dry" ]; then
        echo "replicas=$r scenarios=$n gpus=$((2 * r)) repeat=$REP"
    else
        bash slurm/submit_generate.sh "$r" "$n" "$REP" "$MODEL_DIR" "$C" || echo "skipped replicas=$r scenarios=$n"
    fi
done <<< "$CONFIGS"
