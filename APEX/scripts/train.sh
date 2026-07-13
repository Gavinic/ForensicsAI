#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate codetr
cd src/codetr
./tools/dist_train.sh forgery_configs/co_dino_swinl_m1920.py 1
./tools/dist_train.sh forgery_configs/co_dino_swinl_m1920_ok.py 1

conda activate ms-swift
cd ../../
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model models/Qwen3.5-9B \
    --tuner_type lora \
    --dataset data/LLM/train_v5_redbox.jsonl \
    --val_dataset data/LLM/train_v5_redbox.jsonl \
    --torch_dtype bfloat16 \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --output_dir models/output/Qwen3.5-9B-AIGC \
    --eval_steps 125 \
    --save_steps 125 \
    --save_total_limit 10 \
    --logging_steps 25 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --deepspeed zero2
