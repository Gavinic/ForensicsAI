#!/bin/bash
# Exit immediately on error
set -e

# Auto-detect project root (two levels up)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Switch to the project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "       Starting Stage 2 Final training flow       "
echo "=========================================="

echo ">>> [1/3] Running download_script.sh..."
# Use bash to execute, to avoid issues when the file lacks the executable (x) permission
bash models/download_script.sh

echo ">>> [2/3] Running train_sam.py..."
python src/stage2_final/train_sam.py

echo ">>> [3/3] Running train_LoRA.py..."
python src/stage2_final/train_LoRA.py

echo "=========================================="
echo "                 Training complete                 "
echo "=========================================="
