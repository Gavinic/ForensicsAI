#!/bin/bash
# Exit immediately on error
set -e

# Auto-detect project root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# ==========================================
# Default path configuration
# ==========================================
INPUT_PATH="$PROJECT_ROOT/data/ForgeryAnalysis_Stage_2_Test/Image"
OUTPUT_PATH="$PROJECT_ROOT/results_val/final_submission.csv"

# ==========================================
# Command-line argument parsing
# ==========================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_path)
      INPUT_PATH="$2"
      shift 2  # Move past two arguments (--input_path and its value)
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2  # Move past two arguments (--output_path and its value)
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash inference.sh [--input_path <dir>] [--output_path <file.csv>]"
      exit 1
      ;;
  esac
done

# ==========================================
# Execution flow
# ==========================================
# Switch to the project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "       Starting Stage 2 Final inference flow       "
echo "=========================================="
echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"
echo "=========================================="
echo ">>> [0/2] Running download_script.sh..."
# Use bash to execute, to avoid issues when the file lacks the executable (x) permission
bash models/download_script.sh

echo ">>> [1/2] Running connect-lora.py..."
python src/stage2_final/connect-lora.py

echo ">>> [2/2] Running inference.py..."
# Pass the configured paths as arguments to the Python script
python src/stage2_final/inference.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH"

echo "=========================================="
echo "                Inference complete                "
echo "=========================================="
