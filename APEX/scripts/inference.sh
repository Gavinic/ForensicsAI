#!/bin/bash
INPUT_PATH=$1
OUTPUT_PATH=$2

# check that arguments are provided
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: please provide input path and output path"
    echo "Usage: $0 <input_path> <output_path>"
    exit 1
fi

cd src || { echo "Error: cannot enter src directory"; exit 1; }

# activate conda environment (absolute path or pre-initialized conda recommended)
# initialize conda (if not already initialized)
eval "$(conda shell.bash hook)"

# detection model inference
conda activate codetr
if [ $? -ne 0 ]; then
    echo "Error: cannot activate codetr environment"
    exit 1
fi

# fixed: -output_path should be --output_path
python infer_det.py --input_path "$INPUT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: infer.py execution failed"
    exit 1
fi

# switch to the second environment
conda activate ms-swift
if [ $? -ne 0 ]; then
    echo "Error: cannot activate ms-swift environment"
    exit 1
fi

python infer_llm.py
if [ $? -ne 0 ]; then
    echo "Error: infer_llm.py execution failed"
    exit 1
fi

# uncomment the following if merging is needed
python submit.py --output_path "$OUTPUT_PATH"

echo "All tasks completed"
