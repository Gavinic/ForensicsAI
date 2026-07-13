#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${PROJECT_ROOT}/answer-P50-P51.csv"
OUTPUT_PATH="${PROJECT_ROOT}/answer_with_explanation.csv"
TEST_IMAGE_DIR="${PROJECT_ROOT}/data/ForgeryAnalysis_Stage_1_Test/Image"
LOG_DIR="${PROJECT_ROOT}/stage1_preliminary/logs"
EVIDENCE_JSONL="${PROJECT_ROOT}/data/processed/test_evidence_with_vl.jsonl"
BASE_MODEL_DIR="${PROJECT_ROOT}/models/base/Qwen2.5-14B-Instruct"
VL_MODEL_DIR="${PROJECT_ROOT}/models/base/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="${PROJECT_ROOT}/models/adapters"
ENABLE_VL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_path|--input-path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output_path|--output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --test_image_dir|--test-image-dir)
      TEST_IMAGE_DIR="$2"
      shift 2
      ;;
    --log_dir|--log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --evidence_jsonl|--evidence-jsonl)
      EVIDENCE_JSONL="$2"
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
    --adapter_path|--adapter-path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --disable_vl|--disable-vl)
      ENABLE_VL=0
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_PATH}")" "$(dirname "${EVIDENCE_JSONL}")"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "adapter directory not found: ${ADAPTER_PATH}" >&2
  exit 1
fi

bash "${PROJECT_ROOT}/models/download_script.sh"

CMD=(python "${PROJECT_ROOT}/src/inference.py"
  --input_path "${INPUT_PATH}"
  --output_path "${OUTPUT_PATH}"
  --test_image_dir "${TEST_IMAGE_DIR}"
  --base_model_dir "${BASE_MODEL_DIR}"
  --adapter_path "${ADAPTER_PATH}"
  --vl_model_dir "${VL_MODEL_DIR}"
  --vl_load_in_4bit
  --load_in_4bit
  --gen_batch_size 1
  --evidence_jsonl "${EVIDENCE_JSONL}")

if [[ "${ENABLE_VL}" -eq 1 ]]; then
  CMD+=(--enable_vl)
fi

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/inference.log"
