#!/usr/bin/env bash
set -euo pipefail

# Linux multi-GPU launch script
# Usage: ./scripts/launch_ddp.sh [extra args]

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
CONFIG=${CONFIG:-configs/train_config.yaml}

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m src.train \
  --config "${CONFIG}" \
  "$@"