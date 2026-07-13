#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-<BASE_PATH>/origindata}"
INPUT_PATH="${INPUT_PATH:-$DATA_ROOT/ForgeryAnalysis_Stage_1_Test/Image}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/submission_stage1.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/stage1}"
RUN_NAME="${RUN_NAME:-vit_stage1_pred}"
GPU="${GPU:-0}"
PRED_WEIGHTS="${PRED_WEIGHTS:-$PROJECT_ROOT/outputs/stage1/vit_stage1_ft/best.pth}"
ANSWER_CSV="${ANSWER_CSV:-$PROJECT_ROOT/submission_ensemble.csv}"

bash "$PROJECT_ROOT/models/download_script.sh"

python "$PROJECT_ROOT/src/inference.py" \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --weights_path "$PRED_WEIGHTS" \
  --answer_csv "$ANSWER_CSV" \
  --backbone vit \
  --test_bs 1 \
  --num_workers 8 \
  --gpu "$GPU" \
  "$@"
