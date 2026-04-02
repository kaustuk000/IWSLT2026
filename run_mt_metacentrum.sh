#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -N train-mt
#PBS -l select=1:ncpus=4:ngpus=1:mem=48000mb:gpu_mem=24000mb:scratch_local=100gb
#PBS -l walltime=48:00:00

PROJECT_DIR="${PROJECT_DIR:-$HOME/IWSLT2026-mt}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "$PBS_JOBID running on $(hostname -f), scratch: $SCRATCHDIR" >> "$PROJECT_DIR/jobs_info.txt"

test -n "$SCRATCHDIR" || { echo >&2 "SCRATCHDIR is not set!"; exit 1; }

rsync -a \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'logs' \
    --exclude '__pycache__' \
    "$PROJECT_DIR/" "$SCRATCHDIR/" || { echo >&2 "Failed to copy input files!"; exit 2; }

cd "$SCRATCHDIR" || exit 1

export HF_HOME="${HF_HOME:-$SCRATCHDIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

uv sync

JOB_LOG="$LOG_DIR/mt_${PBS_JOBID}.log"

echo "=== Running MT training ===" | tee -a "$JOB_LOG"
uv run python model/mt_v1.py >> "$JOB_LOG" 2>&1

if [ -d "$SCRATCHDIR/bho_hin_mt" ]; then
    mkdir -p "$PROJECT_DIR/bho_hin_mt"
    rsync -a "$SCRATCHDIR/bho_hin_mt/" "$PROJECT_DIR/bho_hin_mt/" || exit 3
fi

clean_scratch
