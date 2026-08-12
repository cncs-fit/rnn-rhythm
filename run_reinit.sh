#!/bin/bash
# R1-Major1 additional experiment driver.
# Trains 2 conditions x 10 runs with different tau initializations, keeping tau
# learnable. Early-stop goal 1e-5 (justified: tau-amp correlation stable between
# 1e-5 and 1e-6, see reply_to_reviwers/initial_plan.md). Runs are executed with a
# concurrency limit so the single GPU is shared across a few processes.
#
# Output is isolated under multiple_runs_reinit/<condition>/<i>/ and never touches
# the original multiple_runs/.
#
# Usage: bash run_reinit.sh [CONCURRENCY] [GOAL]
set -u

PY=${PYTHON:-python3}
CONC=${1:-4}          # number of concurrent training processes
GOAL=${2:-1e-5}       # early-stopping loss goal
ROOT=multiple_runs_reinit

# job list: "condition seed run_id"
declare -a JOBS
for i in $(seq 1 10); do JOBS+=("loguniform $((1000+i)) $i"); done
for i in $(seq 1 10); do JOBS+=("const200   $((2000+i)) $i"); done

echo "=== R1-1 reinit training: ${#JOBS[@]} jobs, concurrency=$CONC, goal=$GOAL ==="
mkdir -p "$ROOT"

launch() {
    local cond=$1 seed=$2 rid=$3
    local outdir="$ROOT/$cond/$rid"
    mkdir -p "$outdir"
    if [ -f "$outdir/results/checkpoints/best_model.weights.h5" ]; then
        echo "[skip] $cond/$rid already has best_model (done)"
        return
    fi
    echo "[start] cond=$cond seed=$seed run=$rid -> $outdir"
    $PY -u train_reinit.py --seed "$seed" --tau-init "$cond" --goal "$GOAL" \
        --outdir "$outdir/results" > "$outdir/train_log.txt" 2>&1
    echo "[done]  cond=$cond run=$rid  (min_loss: $(grep -o 'min_loss=[0-9.e+-]*' "$outdir/train_log.txt" | tail -1))"
}

running=0
for job in "${JOBS[@]}"; do
    read -r cond seed rid <<< "$job"
    launch "$cond" "$seed" "$rid" &
    running=$((running+1))
    if [ "$running" -ge "$CONC" ]; then
        wait -n
        running=$((running-1))
    fi
done
wait
echo "=== all reinit training jobs finished ==="
