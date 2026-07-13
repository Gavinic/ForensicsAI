#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT=""
OUTPUT_PATH="${PROJECT_DIR}/submission_ensemble.csv"
PRED_DIR="${PROJECT_DIR}/logs/stage1_preliminary/pred"
LOG_DIR="${PROJECT_DIR}/logs/stage1_preliminary"
LOG_PREFIX="inference"
SAM_CHECKPOINT="${PROJECT_DIR}/models/sam_vit_h_4b8939.pth"
ADAPTER_DIR="${PROJECT_DIR}/models/adapters/stage1_preliminary/weight"
THRESHOLD="0.5"
MODE="ensemble"
SINGLE_WEIGHT=""
FOLDS=(0 1 2 3)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_path)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --pred_dir)
      PRED_DIR="$2"
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
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --folds)
      IFS=' ' read -r -a FOLDS <<< "$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --forensics_weights)
      SINGLE_WEIGHT="$2"
      MODE="single"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Usage: bash scripts/stage1_preliminary/inference.sh --input_path /path/to/data_root --output_path /path/to/output.csv" >&2
  exit 1
fi

bash "${PROJECT_DIR}/models/download_script.sh"

mkdir -p "$PRED_DIR"
mkdir -p "$LOG_DIR"

if [[ "$MODE" == "single" ]]; then
  if [[ -z "$SINGLE_WEIGHT" ]]; then
    SINGLE_WEIGHT="${ADAPTER_DIR}/training_kfold_1/forensics_stage1_best.pth"
  fi

  THRESHOLD_FILE="${SINGLE_WEIGHT%.pth}_threshold.json"
  LOG_FILE="${LOG_DIR}/${LOG_PREFIX}_single.log"
  CMD=(python "${PROJECT_DIR}/src/stage1_preliminary/inference.py" \
    --data-root "$DATA_ROOT" \
    --forensics-weights "$SINGLE_WEIGHT" \
    --sam-type vit_h \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --output-csv "$OUTPUT_PATH" \
    --log-file "$LOG_FILE" \
  )
  if [[ -f "$THRESHOLD_FILE" ]]; then
    CMD+=(--threshold-file "$THRESHOLD_FILE")
  else
    CMD+=(--threshold "$THRESHOLD")
  fi
  CMD+=("${EXTRA_ARGS[@]}")
  "${CMD[@]}"
  exit 0
fi

PROB_CSVS=()
THRESHOLD_FILES=()
for fold in "${FOLDS[@]}"; do
  FOLD_NUMBER=$((fold + 1))
  WEIGHT_PATH="${ADAPTER_DIR}/training_kfold_${FOLD_NUMBER}/forensics_stage1_best.pth"
  THRESHOLD_FILE="${ADAPTER_DIR}/training_kfold_${FOLD_NUMBER}/forensics_stage1_best_threshold.json"

  FOLD_OUTPUT="${PRED_DIR}/fold${FOLD_NUMBER}_submission_pred.csv"
  LOG_FILE="${LOG_DIR}/${LOG_PREFIX}_fold${FOLD_NUMBER}.log"
  CMD=(python "${PROJECT_DIR}/src/stage1_preliminary/inference.py" \
    --data-root "$DATA_ROOT" \
    --forensics-weights "$WEIGHT_PATH" \
    --sam-type vit_h \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --output-csv "$FOLD_OUTPUT" \
    --log-file "$LOG_FILE" \
  )
  if [[ -f "$THRESHOLD_FILE" ]]; then
    CMD+=(--threshold-file "$THRESHOLD_FILE")
    THRESHOLD_FILES+=("$THRESHOLD_FILE")
  else
    CMD+=(--threshold "$THRESHOLD")
  fi
  CMD+=("${EXTRA_ARGS[@]}")
  "${CMD[@]}"

  PROB_CSVS+=("${FOLD_OUTPUT%.csv}_prob.csv")
done

ENSEMBLE_LOG_FILE="${LOG_DIR}/${LOG_PREFIX}_ensemble.log"
CMD=(python "${PROJECT_DIR}/src/stage1_preliminary/ensemble_submission.py" \
  --prob-csvs "${PROB_CSVS[@]}" \
  --template-csv "${DATA_ROOT}/submission.csv" \
  --output-csv "$OUTPUT_PATH")
if [[ "${#THRESHOLD_FILES[@]}" -gt 0 ]]; then
  CMD+=(--threshold-files "${THRESHOLD_FILES[@]}")
else
  CMD+=(--threshold "$THRESHOLD")
fi
"${CMD[@]}" 2>&1 | tee "$ENSEMBLE_LOG_FILE"
