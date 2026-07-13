
## Training
# python3 training/train.py --task_target   biomConv_dinov3  --detector_path training/config/detector/biom_conv.yaml
# Detection model training (actual training used a CL approach with multi-step fine-tuning)
export CUDA_VISIBLE_DEVICES=2
python3 ../src/detection/training/train.py --task_target   idmbvit_HT-16  --detector_path ../src/detection/training/config/detector/idbm_vit.yaml


## Fine-tuning based on the llama_factory framework

llamafactory-cli train ../src/llmexp/LlamaFactory/examples/train_lora/qwen3vl_lora_sft_dici.yaml

## RL training based on the SFT model (no clear effect; not used in final result)
python3 ../src/llmexp/grpo_vl.py
