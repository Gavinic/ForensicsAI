#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/explainer_qlora_14b"
LOG_DIR="${PROJECT_ROOT}/stage1_preliminary/logs"
BASE_MODEL_DIR="${PROJECT_ROOT}/models/base/Qwen2.5-14B-Instruct"
VL_MODEL_DIR="${PROJECT_ROOT}/models/base/Qwen2.5-VL-7B-Instruct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root|--data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --processed_dir|--processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --output_dir|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log_dir|--log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --base_model_dir|--base-model-dir)
      BASE_MODEL_DIR="$2"
      shift 2
      ;;
    --vl_model_dir|--vl-model-dir)
      VL_MODEL_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}" "${PROCESSED_DIR}" "${OUTPUT_DIR}"

bash "${PROJECT_ROOT}/models/download_script.sh"

python "${PROJECT_ROOT}/src/data_process.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${PROCESSED_DIR}" \
  --enable-vl \
  --vl-model-name-or-path "${VL_MODEL_DIR}" \
  --vl-load-in-4bit \
  --vl-max-regions 3 \
  --vl-context-ratio 0.35 \
  --vl-min-crop-size 224 \
  --vl-batch-size 8 \
  2>&1 | tee "${LOG_DIR}/prepare_sft.log"

python "${PROJECT_ROOT}/src/train.py" \
  --model-name-or-path "${BASE_MODEL_DIR}" \
  --train-jsonl "${PROCESSED_DIR}/sft_train.jsonl" \
  --val-jsonl "${PROCESSED_DIR}/sft_val.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --load-in-4bit \
  --train-batch-size 1 \
  --grad-accum 8 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --max-length 2048 \
  --gradient-checkpointing \
  --seed 42 \
  2>&1 | tee "${LOG_DIR}/train.log"
