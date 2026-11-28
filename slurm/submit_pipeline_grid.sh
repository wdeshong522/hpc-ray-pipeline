#!/bin/bash
# Usage: slurm/submit_pipeline_grid.sh REPEAT [--dry]
# Weak (N = 1000 K) and strong (N = 4000) pipeline runs at K = 1, 2, 3.
# Order packs the 14-GPU cap: K=3 alone (12), K=2 with K=1 (8 + 4).
# Repeat 1 runs K=3 weak first and the other five runs start only if it succeeds (set SKIP_CANARY=1 to skip).
set -euo pipefail

REP=$1
DRY=${2:-}

submit() {
    if [ "$DRY" = "--dry" ]; then
        echo "would submit: ${DEP:+DEP=$DEP }slurm/submit_pipeline.sh $1 $2 $REP" >&2
        echo "dry"
    else
        slurm/submit_pipeline.sh "$1" "$2" "$REP" || echo "skipped" >&2
    fi
}

CANARY=""
if [ "$REP" = "1" ] && [ "${SKIP_CANARY:-0}" != "1" ]; then
    CANARY=$(DEP= submit 3 3000 | tail -n1)
    echo "canary K=3 weak: $CANARY"
fi

export DEP=$CANARY
for RUN in "3 4000" "2 4000" "1 4000" "2 2000" "1 1000"; do
    set -- $RUN
    ID=$(submit "$1" "$2" | tail -n1)
    echo "K=$1 N=$2 rep$REP: $ID"
done
if [ "$REP" != "1" ] || [ "${SKIP_CANARY:-0}" = "1" ]; then
    DEP= ID=$(DEP= submit 3 3000 | tail -n1)
    echo "K=3 N=3000 rep$REP: $ID"
fi
