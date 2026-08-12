#!/bin/bash
# Run train_once.py multiple times with different random seeds.
# Trained models are collected under multiple_runs/{1..N}/results/.
#
# Usage: ./run_multiple.sh [NUM_RUNS] [BASE_SEED] [START_RUN]
#   NUM_RUNS  : number of training runs (default 20)
#   BASE_SEED : base random seed (default 42); seed for run i is BASE_SEED + i
#   START_RUN : index of the first run (default 1)

set -e

NUM_RUNS=${1:-20}
BASE_SEED=${2:-42}
START_RUN=${3:-1}

END_RUN=$((START_RUN + NUM_RUNS - 1))

echo "Starting $NUM_RUNS runs ($START_RUN..$END_RUN, base_seed=$BASE_SEED)..."

mkdir -p multiple_runs

for i in $(seq $START_RUN $END_RUN); do
    echo "----------------------------------------"
    echo "Run $i of $END_RUN"
    echo "----------------------------------------"

    mkdir -p results

    SEED=$((BASE_SEED + i))
    echo "Seed for this run: $SEED"

    python3 -u train_once.py --seed $SEED 2>&1 | tee train_log.txt

    SAVE_DIR="multiple_runs/$i"
    if [ -d "$SAVE_DIR" ]; then
        echo "Removing existing $SAVE_DIR..."
        rm -rf "$SAVE_DIR"
    fi
    mkdir -p "$SAVE_DIR"

    mv train_log.txt "$SAVE_DIR/"
    if [ -d "results" ]; then
        mv results "$SAVE_DIR/"
    else
        echo "Warning: 'results' directory was not found."
    fi

    echo "Run $i completed."
    echo ""
done

echo "All runs finished."
