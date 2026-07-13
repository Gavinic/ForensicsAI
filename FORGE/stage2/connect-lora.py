import argparse
import os
import shutil

import torch
from peft import PeftModel
from unsloth import FastVisionModel

# ==========================================
# Dynamic path resolution and argument configuration
# ==========================================
# 1. Get the absolute path of the directory containing this script (src/stage2_final)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Locate the project root directory (go up two levels)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# 3. Command-line argument parsing (fixed default paths, supports external arguments)
parser = argparse.ArgumentParser(
    description="Script to physically merge LoRA with the base model"
)
parser.add_argument(
    "--lora_path",
    type=str,
    default=os.path.join(
        PROJECT_ROOT,
        "models",
        "adapters",
        "stage2_final",
        "outputs_qwen3.5_forgery",
        "checkpoint-189",
    ),
    help="Directory where LoRA weights are located",
)
parser.add_argument(
    "--base_model",
    type=str,
    # Core change: move the default expected path of the base model under the models/ folder to keep the root directory clean
    default=os.path.join(PROJECT_ROOT, "models", "Qwen3.5-9B"),
    help="Directory of the clean base model",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=os.path.join(PROJECT_ROOT, "models", "final_full_model"),
    help="Directory to save the merged full model",
)
args = parser.parse_args()

LORA_PATH = args.lora_path
BASE_MODEL = args.base_model
SAVE_PATH = args.save_path

# Print paths for easy debugging and verification
print(f"Project root: {PROJECT_ROOT}")
print(f"LoRA path: {LORA_PATH}")
print(f"Base model: {BASE_MODEL}")
print(f"Save path: {SAVE_PATH}\n")

print("Starting low-level merge process...")

# 1. Load the base model cleanly (must NOT enable 4-bit)
print("1. Loading base model (bfloat16 mode, requires about 18GB memory)...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=False,  # Must be False; high precision is required for the merge addition
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

# 2. Attach the checkpoint patch
print(f"2. Attaching checkpoint: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)

# 3. Physical merge
print("3. Performing low-level parameter welding (merge_and_unload)...")
# This adds the LoRA parameter matrices directly onto the base model matrices and destroys the LoRA structure
model = model.merge_and_unload()

# 4. Native save
print(
    f"4. Writing full weights to: {SAVE_PATH} (disk is spinning hard, please be patient...)"
)
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH, safe_serialization=True)
tokenizer.save_pretrained(SAVE_PATH)

# 5. Fill in the vision config files required by vLLM
print("5. Filling in vision config files...")
for f in ["preprocessor_config.json", "generation_config.json", "config.json"]:
    src = os.path.join(BASE_MODEL, f)
    dst = os.path.join(SAVE_PATH, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

print(
    "\nPhysical merge complete! Check the folder for the multi-GB .safetensors files!"
)
