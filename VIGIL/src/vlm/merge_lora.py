import os

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def merge_lora_weights():
    # ==========================================
    # 1. Path configuration (replace with your actual absolute paths)
    # ==========================================
    # Original main model path (the base after SFT merging)
    base_model_path = "../models/Qwen3-VL-8B-Instruct"

    # Your GRPO LoRA weight path (the checkpoint folder)
    lora_adapter_path = "../models/qwen3-vl-8b_v3/lora/sft"

    # New path where the final merged full model will be saved
    output_merged_path = "../models/qwen3_vl_8b_v3_sft_merged_dici"

    print(f"1. Loading the base model: {base_model_path}")
    # ==========================================
    # 2. Load the base model and Processor
    # ==========================================
    # Note: when merging weights, it is recommended to load on CPU (device_map="cpu") to avoid GPU OOM
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,  # keep the same precision as during training
        device_map="cpu",  # safest to merge in system memory
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"2. Loading LoRA weights: {lora_adapter_path}")
    # ==========================================
    # 3. Load the PEFT model (attach LoRA to the base)
    # ==========================================
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("3. Performing the mathematical merge (merge_and_unload)...")
    # ==========================================
    # 4. Core: multiply the LoRA matrices back into the original model weights
    # ==========================================
    # This step adds the product of Lora_A and Lora_B to the original Linear layers
    merged_model = model_with_lora.merge_and_unload()

    print(f"4. Saving the full merged model to: {output_merged_path}")
    # ==========================================
    # 5. Save the final model and config files
    # ==========================================
    os.makedirs(output_merged_path, exist_ok=True)

    # Save the model weights (defaults to the safest safetensors format)
    merged_model.save_pretrained(output_merged_path, safe_serialization=True)

    # Also save the processor, tokenizer, and all other config files
    processor.save_pretrained(output_merged_path)

    print(" Merge complete!")


if __name__ == "__main__":
    merge_lora_weights()
