cd ../src/hw

conda activate python310

python -u max_vit_forgery_fake.py

python -u max_vit_base_forgery_fake.py

python -u max_vit_large_forgery_fake.py

echo "Classification model training complete. Logs are saved in the logs directory and weights in the output directory."

cd ../Co-DETR

cp instances_tampered_20260302_180354.json ../../data/ForgeryAnalysis_Stage_1_Train/Black/

CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_train.sh cfgs/chuangai/co_dino_vit_large_coco_instance_1280.py 4 ./work_dirs/chuangai/co_dino_vit_large_coco_instance_1280

CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_train.sh cfgs/chuangai/co_dino_vit_large_coco_instance_1280_v2.py 4 ./work_dirs/chuangai/co_dino_vit_large_coco_instance_1280_v2

echo "Segmentation model training complete. Training outputs and weights are saved in the work_dirs directory."

cd ../Forgery

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model ../../models/Qwen3.5-9B \
    --tuner_type lora \
    --dataset './qwen3vl_train_data_v8.jsonl' \
    --val_dataset './qwen3vl_train_data_v8_test.jsonl' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --group_by_length true \
    --output_dir output/Qwen3.5-9B \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --model_author swift \
    --model_name swift-robot

echo "Multimodal model training complete. Training outputs and weights are saved in output/Qwen3.5-9B directory."
