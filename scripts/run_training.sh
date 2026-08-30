#!/usr/bin/env bash
#
# End-to-end training pipeline for a GPU server: install deps, download
# tokenizer models, sanity-check the package, build the vocab, then train.
#
# Usage:
#   ./scripts/run_training.sh                                  # defaults, single GPU/CPU
#   CONFIG=configs/multi30k.yaml ./scripts/run_training.sh
#   DISTRIBUTED=1 ./scripts/run_training.sh                     # all visible GPUs (DDP)
#   FORCE_REBUILD_VOCAB=1 ./scripts/run_training.sh             # rebuild vocab cache
#   SKIP_SANITY_CHECK=1 ./scripts/run_training.sh               # skip the sanity check step
#
# All steps log to stdout AND to logs/run_training_<timestamp>.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/multi30k.yaml}"
DISTRIBUTED="${DISTRIBUTED:-0}"
FORCE_REBUILD_VOCAB="${FORCE_REBUILD_VOCAB:-0}"
SKIP_SANITY_CHECK="${SKIP_SANITY_CHECK:-0}"

mkdir -p logs checkpoints
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="logs/run_training_${TIMESTAMP}.log"

# Mirror everything to a run log so a disconnected SSH session doesn't lose output.
exec > >(tee -a "$RUN_LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== annotated_transformer training pipeline ==="
log "Repo root:   $REPO_ROOT"
log "Config:      $CONFIG"
log "Distributed: $DISTRIBUTED"
log "Run log:     $RUN_LOG"

command -v uv >/dev/null 2>&1 || {
    log "ERROR: 'uv' is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

log "--- Step 1/5: Installing dependencies (uv sync) ---"
uv sync

log "--- Step 2/5: Downloading spaCy tokenizer models ---"
uv run python -m spacy download de_core_news_sm
uv run python -m spacy download en_core_web_sm

if [ "$SKIP_SANITY_CHECK" != "1" ]; then
    log "--- Step 3/5: Running sanity check ---"
    uv run scripts/sanity_check.py
else
    log "--- Step 3/5: Skipping sanity check (SKIP_SANITY_CHECK=1) ---"
fi

log "--- Step 4/5: Building / loading vocabulary ---"
VOCAB_ARGS=(--config "$CONFIG")
if [ "$FORCE_REBUILD_VOCAB" = "1" ]; then
    VOCAB_ARGS+=(--force)
fi
uv run scripts/build_vocab.py "${VOCAB_ARGS[@]}"

log "--- Step 5/5: Training ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU(s) detected:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
else
    log "No nvidia-smi found -- training will run on CPU (expect this to be slow)."
fi

TRAIN_ARGS=(--config "$CONFIG")
if [ "$DISTRIBUTED" = "1" ]; then
    TRAIN_ARGS+=(--distributed)
fi
uv run scripts/train.py "${TRAIN_ARGS[@]}"

log "=== Training pipeline complete ==="
log "Checkpoints: checkpoints/"
log "Metrics:     logs/*.jsonl"
log "Run log:     $RUN_LOG"
