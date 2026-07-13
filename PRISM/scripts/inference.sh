#!/bin/bash

# Initialize variables
INPUT_PATH=""
OUTPUT_PATH=""

# Parse long arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check that required arguments are provided
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: $0 --input_path <input_path> --output_path <output_path>"
    exit 1
fi

echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"

cd src/hw

which conda
source ~/.bashrc

conda activate python310

python -u make_test_data.py --input_path "$INPUT_PATH"

python -u infer_maxvit_base.py

python -u infer_maxvit_large.py

python -u infer_maxvit.py

python -u model_toupiao.py

python -u submit_forgery.py

echo "Classification model inference complete!"

conda activate codetr

cd ../Co-DETR

python -u make_coco_test_data.py --input_path "$INPUT_PATH"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh cfgs/chuangai/co_dino_vit_large_coco_instance_1280.py \
    ../../models/co-detr/stage1-90ea2863.pth 4 \
    --format-only --eval-options "jsonfile_prefix=competetion/chuangai/co_dino_vit_large_coco_instance_1280"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh cfgs/chuangai/co_dino_vit_large_coco_instance_1280_v2.py \
    ../../models/co-detr/stage2-898db43f.pth 4 \
    --format-only --eval-options "jsonfile_prefix=competetion/chuangai/co_dino_vit_large_coco_instance_1280_v2"

cd competetion/chuangai

python -u merge_result_2.py

python -u submit.py

echo "Segmentation model inference complete!"

cd ../../../Forgery

conda activate qwen3.5

python -u make_testonly_data.py --input_path "$INPUT_PATH"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MASTER_PORT=29559 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters ../../models/Qwen3.5-9B-Turn/v1-20260321-220639/checkpoint-310 \
    --val_dataset ./qwen3vl_testBonly.json \
    --enable_thinking false \
    --max_new_tokens 4096 \
    --max_batch_size 16

python -u submit_only_ex.py

cd ../

python -u merge_csv_forgery.py --output_path "$OUTPUT_PATH"

echo "All inference complete. Result saved at Output path: $OUTPUT_PATH"
