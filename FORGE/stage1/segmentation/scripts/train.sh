#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-<BASE_PATH>/origindata}"
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/train.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/stage1}"
RUN_NAME="${RUN_NAME:-vit_stage1_ft}"
GPU="${GPU:-0}"
INIT_WEIGHTS="${INIT_WEIGHTS:-$PROJECT_ROOT/FOCAL_ViT_weights.pth}"

python "$PROJECT_ROOT/src/train.py" \
  --data_root "$DATA_ROOT" \
  --train_csv "$TRAIN_CSV" \
  --backbone vit \
  --weights_path "$INIT_WEIGHTS" \
  --epochs 8 \
  --train_bs 1 \
  --test_bs 1 \
  --num_workers 8 \
  --val_ratio 0.1 \
  --gpu "$GPU" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  "$@"
