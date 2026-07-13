import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re

from datasets import load_dataset
from my_reward import grpo_reward_fn
from peft import LoraConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

model_path = (
    "<BASE_PATH>/data/2026forgery/LlamaFactory/saves/qwen3_vl_8b_v3_sft_merged_dici"
)

MAX_PIXELS = 2048 * 2048
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # <--- key speedup parameter
)
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=True,
    max_pixels=MAX_PIXELS,
)
# print(processor.chat_template)
processor.chat_template = (
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "\n"
    # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' and item['image'] is not none %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif item['type'] == 'text' and item['text'] is not none %}"
    "{{ item['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)
processor.save_pretrained(model_path)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules="all-linear",
    exclude_modules=".*visual.*",  # freeze the visual encoder
    task_type="CAUSAL_LM",
)


output_dir = "./qwen3-vl-grpo-lora3"
dataset = load_dataset(
    "json", data_files="<BASE_PATH>/data/2026forgery/train_dici_trl.json", split="train"
)


# 2. Very important: remove old columns that may interfere with template parsing
# Prevent GRPOTrainer from finding the messages column and applying the template again on its own
# dataset = dataset.remove_columns(["messages"])

print("==before modification==", dataset[0])


training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,  # the learning rate for RL is usually set very small
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=1000,  # adjust according to your dataset size
    per_device_train_batch_size=1,  # high GPU memory usage, keep batch size as small as possible
    gradient_accumulation_steps=5,
    # --- Key vLLM config ---
    # use_vllm=True,                     # enable vLLM to accelerate sampling
    # vllm_mode='colocate',
    # vllm_max_model_length=26334,
    # # vllm_enable_sleep_mode=True,
    # vllm_gpu_memory_utilization=0.5,   # core: reserve 40% GPU memory for vLLM, the rest for training
    # vllm_config={
    #     "trust_remote_code": True,
    #     # pass the in-memory string template directly to vLLM's init parameters
    #     "chat_template": processor.chat_template,
    # },
    # Core GRPO parameters
    num_generations=5,  # number of sampled responses generated per prompt (G=4)
    max_completion_length=966,  # maximum model output length (set to 1024 or higher if long reasoning is needed)
    # max_prompt_length=512,        # maximum prompt length
    temperature=0.9,  # raise the sampling temperature to increase exploration diversity
    beta=0.001,  # KL divergence penalty coefficient, prevents the model from drifting too far from the original weights; usually 0.001
    shuffle_dataset=True,
    bf16=True,
    report_to="tensorboard",
    ## Method type
    loss_type="luspo",  ## luspo (computes loss at the sentence level)
    importance_sampling_level="sequence",  #
    save_steps=100,
)

# Initialize the Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[
        grpo_reward_fn
    ],  # pass multiple reward functions; the scores are summed automatically
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)
print("Starting GRPO LoRA training...")
trainer.train()
