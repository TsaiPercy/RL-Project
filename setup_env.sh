#!/usr/bin/env bash
# Usage: source setup_env.sh
# Sets all ML-related caches to project-local directories.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="$PROJECT_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/huggingface/hub"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/huggingface/datasets"
export TORCH_HOME="$PROJECT_DIR/cache/torch"
export WANDB_DIR="$PROJECT_DIR/cache/wandb"
export WANDB_CACHE_DIR="$PROJECT_DIR/cache/wandb"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$WANDB_DIR"
echo "Project cache configured: $PROJECT_DIR/cache/"
