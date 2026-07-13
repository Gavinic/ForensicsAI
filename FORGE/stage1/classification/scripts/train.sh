#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=""
OUTPUT_DIR="${PROJECT_DIR}/models/adapters/stage1_preliminary/weight"
LOG_DIR="${PROJECT_DIR}/logs/stage1_preliminary"
LOG_PREFIX="train"
SAM_CHECKPOINT="${PROJECT_DIR}/models/sam_vit_h_4b8939.pth"
INIT_FORENSICS_WEIGHTS="${PROJECT_DIR}/models/forgery_experts.pth"
NUM_FOLDS="5"
THRESHOLD="0.5"
FOLDS=(0 1 2 3)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_path|--data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log_prefix)
      LOG_PREFIX="$2"
      shift 2
      ;;
    --num_folds)
      NUM_FOLDS="$2"
      shift 2
      ;;
    --folds)
      IFS=' ' read -r -a FOLDS <<< "$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Usage: bash scripts/stage1_preliminary/train.sh --input_path /path/to/data_root [--output_dir /path/to/output_dir]" >&2
  exit 1
fi

bash "${PROJECT_DIR}/models/download_script.sh"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

for fold in "${FOLDS[@]}"; do
  FOLD_NUMBER=$((fold + 1))
  FOLD_OUTPUT_DIR="${OUTPUT_DIR}/training_kfold_${FOLD_NUMBER}"
  LOG_FILE="${LOG_DIR}/${LOG_PREFIX}_fold${FOLD_NUMBER}.log"
  mkdir -p "$FOLD_OUTPUT_DIR"
  python "${PROJECT_DIR}/src/stage1_preliminary/train_sam.py" \
    --data-root "$DATA_ROOT" \
    --sam-type vit_h \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --init-forensics-weights "$INIT_FORENSICS_WEIGHTS" \
    --output-dir "$FOLD_OUTPUT_DIR" \
    --best-model-name "forensics_stage1_best.pth" \
    --last-model-name "forensics_stage1_last.pth" \
    --num-folds "$NUM_FOLDS" \
    --fold-index "$fold" \
    --threshold "$THRESHOLD" \
    --log-file "$LOG_FILE" \
    "${EXTRA_ARGS[@]}"
done
