#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -N train-e2e
#PBS -l select=1:cluster=bee*|zia*:ncpus=4:ngpus=1:mem=64000mb:gpu_mem=40000mb:scratch_local=100gb
#PBS -l walltime=48:00:00

PROJECT_DIR="${PROJECT_DIR:-$HOME/IWSLT2026-e2e}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "$PBS_JOBID running on $(hostname -f), scratch: $SCRATCHDIR" >> "$PROJECT_DIR/jobs_info.txt"

test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }
test -n "$HF_TOKEN" || { echo >&2 "HF_TOKEN is not set!"; exit 2; }

rsync -a \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'logs' \
    --exclude '__pycache__' \
    "$PROJECT_DIR/" "$SCRATCHDIR/" || { echo >&2 "Failed to copy input files!"; exit 3; }

cd "$SCRATCHDIR" || exit 1

export HF_HOME="${HF_HOME:-$SCRATCHDIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

uv sync

JOB_LOG="$LOG_DIR/e2e_${PBS_JOBID}.log"

echo "=== Running e2e training ===" | tee -a "$JOB_LOG"
bash run.sh >> "$JOB_LOG" 2>&1

if [ -d "$SCRATCHDIR/checkpoints" ]; then
    mkdir -p "$PROJECT_DIR/checkpoints"
    rsync -a "$SCRATCHDIR/checkpoints/" "$PROJECT_DIR/checkpoints/" || exit 4
fi
if [ -f "$SCRATCHDIR/train_output.log" ]; then
    cp "$SCRATCHDIR/train_output.log" "$PROJECT_DIR/train_output.log"
fi

clean_scratch
