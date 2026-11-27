#!/bin/bash
# Usage: slurm/submit_judge_grid.sh REPEAT [--dry]
# First a 100-response placement test (2 units, 8 GPUs); the grid starts only if it succeeds.
# Grid order packs the 14-GPU cap: 12 + 2, then 8 + 4 (+ the next repeat's 2).
# Set SKIP_TEST=1 to skip the test (repeats 2 and 3).
set -euo pipefail

REP=$1
DRY=${2:-}
N=4000

submit() {
    if [ "$DRY" = "--dry" ]; then
        echo "would submit: ${DEP:+DEP=$DEP }slurm/submit_judge.sh $1 $2 $3" >&2
        echo "dry"
    else
        slurm/submit_judge.sh "$1" "$2" "$3" || echo "skipped" >&2
    fi
}

TEST_ID=""
if [ "${SKIP_TEST:-0}" != "1" ]; then
    TEST_ID=$(DEP= submit u2 100 0 | tail -n1)
    echo "placement test job: $TEST_ID"
fi

export DEP=$TEST_ID
for MODE in u3 serial u2 u1; do
    ID=$(submit "$MODE" "$N" "$REP" | tail -n1)
    echo "$MODE rep$REP job: $ID"
done
