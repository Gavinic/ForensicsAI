#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH=""
OUTPUT_PATH=""
RAW_OUTPUT_PATH=""
MASK_OUTPUT_DIR=""
CHECKPOINT_PATH=""
BASE_MODEL_PATH=""
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_DOWNLOAD=0

usage() {
  cat <<EOF
Usage: bash scripts/inference.sh --input_path <test_dir> --output_path <submission.csv|submission.json> [options]

Required arguments:
  --input_path        Test-set root directory or the image directory itself.
  --output_path       Final prediction file path (.csv or .json).

Optional arguments:
  --raw_output_path   Intermediate raw inference CSV path. Defaults next to output_path.
  --mask_output_dir   Optional directory for decoded binary masks.
  --checkpoint_path   LoRA checkpoint directory. Defaults to the best available local checkpoint.
  --base_model_path   Base model directory. Defaults to models/Qwen3-VL-8B-Instruct.
  --python_bin        Python interpreter used to run the two Python scripts.
  --skip_download     Skip models/download_script.sh.
  -h, --help          Show this help message.
EOF
}

pick_default_checkpoint() {
  local candidates=(
    "${PROJECT_ROOT}/models/adapters/stage2_final/checkpoint-4200"
    "${PROJECT_ROOT}/models/adapters/stage1_preliminary/checkpoint-2600"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/adapter_config.json" && -f "${candidate}/adapter_model.safetensors" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "${candidates[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_path)
      INPUT_PATH="${2:-}"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --raw_output_path)
      RAW_OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --mask_output_dir)
      MASK_OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="${2:-}"
      shift 2
      ;;
    --base_model_path)
      BASE_MODEL_PATH="${2:-}"
      shift 2
      ;;
    --python_bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --skip_download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "$INPUT_PATH" ]] || { echo "Missing --input_path" >&2; usage >&2; exit 1; }
[[ -n "$OUTPUT_PATH" ]] || { echo "Missing --output_path" >&2; usage >&2; exit 1; }

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No usable Python interpreter found." >&2
    exit 1
  fi
fi

if [[ -z "$BASE_MODEL_PATH" ]]; then
  BASE_MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct"
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_PATH="$(pick_default_checkpoint)"
fi

if [[ -z "$RAW_OUTPUT_PATH" ]]; then
  output_dir="$(dirname "$OUTPUT_PATH")"
  mkdir -p "$output_dir"
  output_dir="$(cd "$output_dir" && pwd)"
  output_name="$(basename "$OUTPUT_PATH")"
  output_stem="${output_name%.*}"
  RAW_OUTPUT_PATH="${output_dir}/${output_stem}.raw_infer.csv"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$RAW_OUTPUT_PATH")"
if [[ -n "$MASK_OUTPUT_DIR" ]]; then
  mkdir -p "$MASK_OUTPUT_DIR"
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  MODELS_ROOT="${PROJECT_ROOT}/models" \
  BASE_MODEL_DIR="$BASE_MODEL_PATH" \
  BASE_MODEL_INFO="${PROJECT_ROOT}/models/base_model_info.json" \
  CHECKPOINT_DIR="$CHECKPOINT_PATH" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash "${PROJECT_ROOT}/models/download_script.sh"
fi

"$PYTHON_BIN" "${PROJECT_ROOT}/src/inference-stage2.py" \
  --input_path "$INPUT_PATH" \
  --output_path "$RAW_OUTPUT_PATH" \
  --base_model_path "$BASE_MODEL_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH"

norm_cmd=(
  "$PYTHON_BIN"
  "${PROJECT_ROOT}/src/norm-checkpoint2submission.py"
  --input_path "$RAW_OUTPUT_PATH"
  --dataset_path "$INPUT_PATH"
  --output_path "$OUTPUT_PATH"
)

if [[ -n "$MASK_OUTPUT_DIR" ]]; then
  norm_cmd+=(--mask_output_dir "$MASK_OUTPUT_DIR")
fi

"${norm_cmd[@]}"

echo "Raw inference CSV: ${RAW_OUTPUT_PATH}"
echo "Final submission: ${OUTPUT_PATH}"
