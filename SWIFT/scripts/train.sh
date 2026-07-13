#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
LOG_FILE="logs/0329-2_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=0

if ! command -v swift >/dev/null 2>&1; then
  echo "Error: swift command not found; ms-swift may not be installed or the environment is not activated." | tee -a "$LOG_FILE"
  echo "Please run: conda activate <BASE_PATH>" | tee -a "$LOG_FILE"
  exit 1
fi

if ! python -c "import swift" >/dev/null 2>&1; then
  echo "Error: cannot import swift in the current Python environment; ms-swift installation may be incomplete." | tee -a "$LOG_FILE"
  echo "Please confirm the active environment is: <BASE_PATH>" | tee -a "$LOG_FILE"
  exit 1
fi

echo "ms-swift check passed, starting training..." | tee -a "$LOG_FILE"
swift sft --config 0330-1-stage1.yaml 2>&1 | tee -a "$LOG_FILE"
