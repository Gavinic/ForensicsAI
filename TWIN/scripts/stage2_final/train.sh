#!/bin/bash

# Start training
export CUDA_VISIBLE_DEVICES=0  # Ensure only one GPU is visible

# Environment and path configuration
MODEL_PATH="<BASE_PATH>/models/qwen3-vl-8b-instruct"
OUTPUT_DIR="./output/multitask_v5_overfit"
DATASETS="stage_2_multitask_v2"

nohup python ../../src/qwenvl/train/train_qwen.py \
    --model_name_or_path "$MODEL_PATH" \
    --tune_mm_llm True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --model_max_length 2048 \
    --max_pixels 1003520 \
    --num_train_epochs 5 \
    --logging_steps 5 \
    --save_strategy "epoch" \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to "none" \
    --gradient_checkpointing True \
    > nohup_01_multitask_v4.out &


OUTPUT_DIR="./output/lora_b_v1"
DATASETS="lora_b"

nohup python ../../src/qwenvl/train/train_qwen.py \
    --model_name_or_path "$MODEL_PATH" \
    --tune_mm_llm True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --model_max_length 512 \
    --max_pixels 1003520 \
    --num_train_epochs 8 \
    --logging_steps 5 \
    --save_strategy "epoch" \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to "none" \
    --gradient_checkpointing True \
    > nohup_02_lora_b.out &


nohup python ../../src/train_segformer.py \
    --forgery_dir ../datasets/ForgeryAnalysis_Stage_1_Train \
    --data_dir ../datasets/tianchi_2022/train \
    --data_analysis ../datasets/tianchi_2022/data_mask_analysis.json \
    --min_mask_ratio 0.01 \
    --val_split 0.0 \
    --seed 42 \
    --image_size 512 \
    --model b3 \
    --pretrained ../models/segformer_b3_pretrained.pth \
    --batch_size 8 \
    --epochs 50 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --aug_level geometric \
    --use_early_stopping \
    --patience 10 \
    --output_dir models/data_progressive \
    --exp_name None \
    > train_segformer.out &
