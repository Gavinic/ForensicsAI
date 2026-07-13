import glob
import logging
import os
import shutil
import sys
import time

from transformers import TrainerCallback  # Import the callback module

# =================================================================
# 0. Dynamic path resolution and environment variables (must be set before importing torch)
# =================================================================
# 1. Get the absolute path of the directory containing this script (src/stage2_final)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Locate the project root directory (go up two levels)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# 3. Add the project root to the environment variables to prevent ModuleNotFoundError during cross-directory calls (e.g. import src...)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Unified cache path to avoid polluting the global environment
os.environ["MODELSCOPE_CACHE"] = os.path.join(PROJECT_ROOT, ".cache")
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache")

# =================================================================
# 0.5 Global logging configuration (append mode)
# =================================================================
# Path alignment: output directly to the logs/stage2_final directory
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "stage2_final")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
LOG_FILE = os.path.join(LOG_DIR, "train.log")

# Configure the root logger to output to both console and file, using 'a' (append) mode
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("A new round of training task started")
logger.info("=" * 50)


# =================================================================
# 0.6 Custom training log callback (records Loss per step)
# =================================================================
class StepLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called whenever the Trainer fires a log event (controlled by logging_steps).
        """
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            epoch = logs.get("epoch", 0.0)

            # Format the output and write to train.log
            self.logger.info(
                f"[Training] Epoch: {epoch:.4f} | Step: {step} | Loss: {loss:.4f} | LR: {lr:.2e}"
            )


import torch
from datasets import Dataset
from datasets import Image as HfImage
from PIL import Image as PilImage
from PIL import ImageFile
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
PilImage.MAX_IMAGE_PIXELS = None

# =================================================================
# 1. Global configuration (dynamically assembled from project root)
# =================================================================
# [Fix] Base model path: the original code looked directly in the root; per the directory tree this is corrected to look under models/
MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-9B")

# Original training dataset path
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ForgeryAnalysis_Stage_1_Train")

# Cleaning directory: physically downscale 8K large images to prevent VRAM from blowing up (stored under the data directory)
CLEAN_DATA_ROOT = os.path.join(
    PROJECT_ROOT, "data", "ForgeryAnalysis_Stage_1_Train_Cleaned_1024"
)

# LoRA weights output directory (aligned with the models/adapters path in the project structure tree)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "models", "adapters", "stage2_final", "outputs_qwen3.5_forgery"
)

MAX_IMAGE_SIZE = 1024
MAX_SEQ_LENGTH = 4096

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Model path: {MODEL_NAME}")
logger.info(f"Raw data: {RAW_DATA_ROOT}")
logger.info(f"Cleaned data: {CLEAN_DATA_ROOT}")
logger.info(f"Output directory: {OUTPUT_DIR}\n")


# ==========================================
# 2. Physical downscaling and data cleaning (core defense)
# ==========================================
def prepare_cleaned_dataset(raw_root, clean_root, max_size):
    if os.path.exists(clean_root):
        logger.info(
            f"Already-cleaned directory detected {clean_root}; skipping preprocessing."
        )
        return

    logger.info(
        f"Physically downscaling original images to {max_size}px (dimensionality reduction for 8K images)..."
    )
    valid_count = 0
    for cat in ["Black", "White"]:
        img_raw_dir = os.path.join(raw_root, cat, "Image")
        cap_raw_dir = os.path.join(raw_root, cat, "Caption")

        img_clean_dir = os.path.join(clean_root, cat, "Image")
        cap_clean_dir = os.path.join(clean_root, cat, "Caption")

        # Tolerant mode: if the original image folder does not exist, skip this category
        if not os.path.exists(img_raw_dir):
            logger.warning(f"Path not found: {img_raw_dir}; skipping this category")
            continue

        os.makedirs(img_clean_dir, exist_ok=True)
        os.makedirs(cap_clean_dir, exist_ok=True)

        for img_file in os.listdir(img_raw_dir):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue

            raw_path = os.path.join(img_raw_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            md_path = os.path.join(cap_raw_dir, base_name + ".md")

            if not os.path.exists(md_path):
                continue

            try:
                with PilImage.open(raw_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    w, h = img.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        img = img.resize(
                            (int(w * scale), int(h * scale)),
                            resample=PilImage.Resampling.LANCZOS,
                        )
                    img.save(
                        os.path.join(img_clean_dir, base_name + ".jpg"),
                        "JPEG",
                        quality=95,
                    )

                shutil.copy2(md_path, os.path.join(cap_clean_dir, base_name + ".md"))
                valid_count += 1
            except Exception as e:
                logger.warning(f"Skipping abnormal file {img_file}: {e}")

    logger.info(
        f"Data preprocessing complete; generated {valid_count} physically clean records!"
    )


# ==========================================
# 3. Build the Dataset (multiprocess & dual-prompt version)
# ==========================================
def load_forgery_dataset(data_root):
    data_list = []

    # The two prompts requested
    PROMPT_BLACK = "Analyze this image. This is a tampered (forged) image; point out the tampering traces and describe the target:"
    PROMPT_WHITE = "Analyze this image. This is a real, original image; describe the target content in the image in detail:"

    for category in ["Black", "White"]:
        img_dir = os.path.join(data_root, category, "Image")
        cap_dir = os.path.join(data_root, category, "Caption")

        # Assign the corresponding user prompt based on the category
        current_prompt = PROMPT_BLACK if category == "Black" else PROMPT_WHITE

        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.abspath(os.path.join(img_dir, img_file))
            md_path = os.path.join(cap_dir, base_name + ".md")

            if not os.path.exists(md_path):
                continue

            with open(md_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            data_list.append(
                {"image": img_path, "user_prompt": current_prompt, "caption": caption}
            )

    ds = Dataset.from_list(data_list)
    ds = ds.cast_column("image", HfImage())  # Lazy loading to prevent OOM

    def format_messages(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": example["user_prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["caption"]}],
                },
            ]
        }

    # Multiprocess processing; safely remove extra columns to avoid the Arrow bug
    ds = ds.map(format_messages, num_proc=4)
    cols_to_remove = [c for c in ds.column_names if c != "messages"]
    return ds.remove_columns(cols_to_remove)


# ==========================================
# 4. Main training flow
# ==========================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # [Stage 1] Physically clean the large images
    if local_rank == 0:
        prepare_cleaned_dataset(RAW_DATA_ROOT, CLEAN_DATA_ROOT, MAX_IMAGE_SIZE)

    # Distributed barrier to ensure all processes see the cleaned data
    if torch.cuda.device_count() > 1:
        (
            torch.distributed.barrier()
            if torch.distributed.is_initialized()
            else time.sleep(5)
        )

    # [Stage 2] Build the dataset
    logger.info(f"Process {local_rank} is loading the dataset...")
    dataset = load_forgery_dataset(CLEAN_DATA_ROOT)

    # [Stage 3] Load the model
    logger.info(f"Process {local_rank} is loading the model into VRAM...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        local_files_only=True,
        device_map={"": local_rank},
    )

    # Explicitly enable vision training mode
    FastVisionModel.for_training(model)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # For a forensics model, fine-tuning the vision layers is recommended to capture subtle artifacts
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
    )

    # [Stage 4] Start training
    logger.info(
        f"Process {local_rank} configuration complete; starting formal training..."
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="",
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=[
            StepLoggingCallback(logger)
        ],  # Inject the custom callback to record Loss per step
        args=SFTConfig(
            per_device_train_batch_size=4,  # With 96G VRAM + 1024px images, this can be set to 4 or higher
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,  # 3 epochs is the standard for fine-tuning
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,  # Keep at 1 to ensure the callback fires every step
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            seed=3407,
            remove_unused_columns=False,
            dataset_kwargs={
                "skip_prepare_dataset": True
            },  # Key parameter to avoid Unsloth errors
            report_to="none",  # none here means not uploading to wandb/tensorboard; local python logs are still captured
        ),
    )

    trainer.train()

    if local_rank == 0:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Task complete! Weights saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
