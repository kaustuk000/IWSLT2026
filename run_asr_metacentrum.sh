#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -N train-asr
#PBS -l select=1:ncpus=4:ngpus=1:mem=64000mb:gpu_mem=40000mb:scratch_local=100gb
#PBS -l walltime=48:00:00

PROJECT_DIR="${PROJECT_DIR:-$HOME/IWSLT2026-asr}"
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

export WORK_DIR="$SCRATCHDIR"
export HF_HOME="${HF_HOME:-$SCRATCHDIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

uv sync

JOB_LOG="$LOG_DIR/asr_${PBS_JOBID}.log"

echo "=== Running ASR training ===" | tee -a "$JOB_LOG"
uv run python WhisperLv3FT.py >> "$JOB_LOG" 2>&1

mkdir -p "$PROJECT_DIR/whisper-bhojpuri" "$PROJECT_DIR/whisper-bhojpuri-final"
rsync -a "$SCRATCHDIR/whisper-bhojpuri/" "$PROJECT_DIR/whisper-bhojpuri/" || exit 3
if [ -d "$SCRATCHDIR/whisper-bhojpuri-final" ]; then
    rsync -a "$SCRATCHDIR/whisper-bhojpuri-final/" "$PROJECT_DIR/whisper-bhojpuri-final/" || exit 4
fi
for file in results_table.csv epoch_summary.csv; do
    if [ -f "$SCRATCHDIR/$file" ]; then
        cp "$SCRATCHDIR/$file" "$PROJECT_DIR/$file"
    fi
done

clean_scratch
