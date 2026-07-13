import importlib
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# # Fixed User Prompt
# PROMPT_TEXT = """You are a professional image forensics and forgery analysis expert. Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.
# The output JSON must contain the following three fields:
# 1. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local tampering, splicing, etc.).
# 2. "boxes": a 2D list. If the image is judged as forged, provide the list of bounding boxes for all tampered regions, in the format [[x_min, y_min, x_max, y_max], ...], where coordinate values must be integers scaled to the 0-1000 range. If it is a real image, output an empty list [].
# 3. "explanation": a string. Provide a professional, rigorous, and logically clear natural-language forensic report.
# When writing the explanation, be sure to follow these analysis norms:
# - Structure: first give the overall verdict, then elaborate on the reasons, and finally give a concluding statement. For forged images, you must mark the normalized coordinates inline when describing the tampered area (e.g. "located at coordinates [x1, y1, x2, y2]").
# - Visual feature analysis: observe and describe in detail the differences between the target area and the surrounding environment in resolution, pixelation, edge transition (whether hard or with a cutout feel), lighting consistency (environmental reflection, shadow direction), noise distribution, and material texture. For generated people, focus on the hands and face.
# - Logic and common-sense checks: combined with the scene text or specific content in the image, analyze whether it conforms to physical laws, normal optical imaging characteristics (e.g. depth of field), industry common sense (e.g. currency decimal places, brand spelling), logical consistency (e.g. bill amount calculation), layout norms, or business logic (e.g. whether the content has systematic contradictions or violates the reasonableness of the specific application scenario).
# """

PROMPT_TEXT = """You are a top-tier image forensics and forgery analysis expert, especially proficient in tampering detection and AIGC-generation identification for "scene text images (all kinds of receipts, certificates, signs, and natural-scene text)". Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.

[Output requirements]
You must output exactly one valid JSON object containing the following three fields:
1. "boxes": a 2D list. If judged as forged, provide the list of bounding boxes for all tampered/generated anomaly regions, in the format [[x_min, y_min, x_max, y_max], ...]; coordinate values must be integers scaled to the 0-1000 range. If judged as a real image, output an empty list [].
2. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local text tampering, erasure-rewrite, splicing, etc.).
3. "explanation": a string. Provide a professional, rigorous, structured, and logically clear natural-language forensic report.

[Explanation writing spec and reasoning chain]
Be sure to organize the explanation text strictly according to the following four layers:

Layer 1: Overall verdict
- State the nature of the image directly.

Layer 2: Tampering localization and content mapping (Grounding alignment)
- List each key tampered region one by one, and bind the normalized coordinates to the tampered content precisely within the text.
- Format reference: "There are [N] key tampered regions in the image, whose coordinates and content are respectively: [semantic description] [x1, y1, x2, y2] content is '[specific text]'; ...".

Layer 3: Visual and low-level feature analysis (Visual Forensics)
- For receipts/documents focus on: differences between tampered digits and the original text in font weight (bolder/blacker), font style (e.g. whether the decimal point is a square pixel block or round), edge transition (whether it lacks the faint graininess that dot-matrix/thermal printing should have, or is too smooth), and layout alignment (vertical baseline misalignment, unnatural spacing).
- For scene/AIGC focus on: whether text rendering is affected by scene lighting (whether it appears as an independent planar light emitter or floating), whether perspective transformation conforms to the physical bumps of the 3D surface, and whether there is unidentifiable distorted garbled text (AIGC hallucination characters).
- Background and artifact analysis: whether there are unnatural smooth, blurry, or smearing marks around tampered regions (erasure-repair artifacts), and whether the original paper-fiber texture, environmental noise, or JPEG compression ghosting is destroyed. If a human body is involved, point out anatomical distortions (e.g. fused fingers, disproportionate ratios, waxy feel).

Layer 4: Logic and common-sense cross-checking (Logical Consistency)
- Logic computation: verify the mathematical self-consistency of the data inside the receipt (e.g. whether unit price x quantity equals the total, whether the sum of all subtotals equals the grand total, and whether the tax calculation is correct).
- Industry common sense: check whether the text content conforms to the norms of the specific scenario (e.g. Malaysian ringgit should keep two rather than three decimal places, whether specific leading zeros are reasonable, and business tax-rate common sense).
- Physical laws: analyze whether the shadow projection direction matches the light source, whether the support structure conforms to mechanical common sense, and whether reflective surfaces (e.g. water reflections) show continuity contradictions.

Layer 5: Concluding statement
- Briefly describe the possible intent of the forgery (e.g. fabricating transaction records for false reimbursement, creating misleading information, etc.) and give the final untrustworthy conclusion.
"""

# PROMPT_TEXT = """You are a top-tier image forensics and forgery analysis expert, proficient in digital tampering detection (especially scene text, receipts/documents) and in-depth identification of AIGC-generated images. Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.
# [Output requirements]
# You must output exactly one valid JSON object containing the following three fields:
# 1. "boxes": a 2D list. If judged as forged, provide the list of bounding boxes for all tampered/generated anomaly regions, in the format [[x_min, y_min, x_max, y_max], ...]; coordinate values must be integers scaled to the 0-1000 range. If judged as a real image, output an empty list [].
# 2. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local tampering, erasure-rewrite, splicing, etc.).
# 3. "explanation": a string. Provide a professional, rigorous, structured, and logically clear natural-language forensic report.
# [Explanation writing spec and reasoning chain]
# Be sure to organize the explanation text strictly according to the following layered structure, and adopt the corresponding writing logic based on authenticity (label):
# Layer 1: Overall verdict
# - Forged (Label=1): state the conclusion directly (e.g. "This is a forged gas-station receipt produced by digital tampering" or "This is a forged digital image produced by artificial intelligence").
# - Real (Label=0): explicitly confirm authenticity (e.g. "This is a real photo of a shop storefront, with no signs of digital forgery or post-processing tampering").
# Layer 2: Tampering localization and content mapping (only required for forged images)
# - List each key anomaly/tampered region one by one, and bind the normalized coordinates to the content precisely within the text.
# - Format reference: "There are [N] key tampered/forged regions in the image, whose coordinates and content are respectively: [semantic description] [x1, y1, x2, y2] content is '[specific text/structure]'; ...".
# Layer 3: Visual feature and low-level signal analysis
# - For forged features:
#   - Text tampering: unnatural mixing of dot-matrix/thermal fonts with sans-serif smooth fonts; vertical baseline misalignment; abnormal stroke thickness; lack of the "ink-bleed" or "burrs" that real ink should leave on paper; smooth background smearing, pixelated noise breaks, and clone-stamp artifacts left by erasure-repair.
#   - AIGC hallucinations: meaningless repeated consonants (e.g. Swwwwn), unreadable garbled text, characters "melting" or sticking together; "waxy" skin and over-smoothing; "perfect mirror symmetry" that violates real-world tolerances; resolution inversion (e.g. blurry face but sharp hands); floating feel from shadow/light-source mismatch.
# - For real features:
#   - Describe the naturalness of physical interaction: natural perspective deformation of the text baseline following paper folds; consistency of global illumination and environmental reflection; uniformity of digital noise distribution; the real overlay of physical flaws (stains, tears) and ink.
# Layer 4: Logic and common-sense cross-checking
# - Receipt/invoice-specific checklist: perform rigorous mathematical self-consistency and business common-sense checks, focusing on the following common forgery loopholes:
#   1. Abnormal decimal places: check whether the amount violates the common sense of a specific country's currency (e.g. three decimal places in regular retail transactions for Malaysian ringgit, RMB, etc.).
#   2. Tax-rate/amount contradiction: check whether the tax calculation is correct (especially a 0% tax-rate item being wrongly computed with a non-zero tax, or a fixed rate such as 6% not matching the figure on the receipt).
#   3. Arithmetic addition errors: rigorously check whether unit price x quantity equals the item total, whether the sum of all item subtotals equals the subtotal/total, and whether the payment amount and change match.
#   4. Spatio-temporal and date fallacies: check whether the date format is valid (e.g. "month 24", "29/24" and other non-existent dates), and whether the year is anachronistic relative to the policy recorded on the receipt (e.g. GST tax system) or a specific era.
# - AIGC/scene common-sense checks: verify the clarity principle of commercial logos (a blurry or garbled sponsor logo violates business logic); the semantic reasonableness of brand naming; whether the physical structure conforms to mechanical principles (e.g. an unsupported cantilevered building).
# Layer 5: Concluding statement
# - Forged: point out the tampering intent (fabricating transactions, inflating reimbursements, building a false identity, etc.) and state that the document "lacks authenticity and credibility / is inadmissible".
# - Real: summarize the self-consistency of the logic and state that "based on comprehensive analysis, this image truthfully records the original state / the actually existing scene".
# """

PROMPT_TEXT1 = "Locate the tampered region in the image."


# ==================== Dataset class ====================
class ForgeryAnalysisDataset(Dataset):
    """Forgery analysis multimodal dataset"""

    def __init__(self, data_path: str, processor, tokenizer, max_length: int = 4096):
        self.data_path = data_path
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data()
        print(f"Loaded dataset: {len(self.samples)} samples")

    def _load_data(self) -> List[Dict]:
        """Load JSONL data"""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing failed on line {line_num}: {e}")
                    continue
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        messages = sample

        # Extract and load the image
        img_path = messages["image"]
        image = Image.open(img_path).convert("RGB")
        assistant_text = messages["explanation"]
        grounding = messages.get("grounding", None)
        prompt_text = PROMPT_TEXT1
        # Set the maximum number of tokens you expect, e.g. 2048 tokens
        # 3500 * 28 * 28
        MAX_PIXELS = 5000 * 32 * 32
        if grounding is None:
            prompt_text = PROMPT_TEXT
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "max_pixels": MAX_PIXELS,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Use the processor to apply the chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process visual information
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        # Process the input
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            do_resize=False,
        )

        # Prepare labels (mask out the user part)
        instruction_input_ids = inputs["input_ids"][0]
        instruction_attention_mask = inputs["attention_mask"][0]
        # pixel_values = inputs.get("pixel_values", None)
        # image_grid_thw = inputs.get("image_grid_thw", None)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"][0]

        # Tokenize the assistant response
        response = self.tokenizer(assistant_text, add_special_tokens=False)
        response_input_ids = response["input_ids"]
        response_attention_mask = response.get(
            "attention_mask", [1] * len(response_input_ids)
        )

        # Add the EOS token
        eos_token_id = self.tokenizer.eos_token_id
        # if eos_token_id is not None:
        #     if not response_input_ids or response_input_ids[-1] != eos_token_id:
        #         response_input_ids = response_input_ids + [eos_token_id]
        #         response_attention_mask = response_attention_mask + [1]
        if eos_token_id is not None:
            if not response_input_ids or response_input_ids[-1] != eos_token_id:
                response_input_ids = response_input_ids + [eos_token_id]
                response_attention_mask = response_attention_mask + [1]
        else:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError(
                    "Either eos_token_id or pad_token_id must be defined to end the response sequence."
                )
            response_input_ids = response_input_ids + [pad_token_id]
            response_attention_mask = response_attention_mask + [1]

        # Merge input_ids and labels
        input_ids = instruction_input_ids + response_input_ids
        attention_mask = instruction_attention_mask + response_attention_mask
        labels = [-100] * len(instruction_input_ids) + response_input_ids

        # Truncate
        if len(input_ids) > self.max_length:
            print(
                f"Input sequence length exceeds max_length: {len(input_ids)} > {self.max_length}"
            )
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": (
                pixel_values
                if pixel_values is not None
                else torch.zeros(1, 3, 224, 224)
            ),
            "image_grid_thw": (
                image_grid_thw
                if image_grid_thw is not None
                else torch.tensor([[1, 8, 8]])
            ),
        }


# ==================== Data Collator ====================
class QwenVLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_tensors = [
            torch.as_tensor(sample["input_ids"], dtype=torch.long)
            for sample in features
        ]
        attention_tensors = [
            torch.as_tensor(sample["attention_mask"], dtype=torch.long)
            for sample in features
        ]
        label_tensors = [
            torch.as_tensor(sample["labels"], dtype=torch.long) for sample in features
        ]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else self.tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError(
                "Both pad_token_id and eos_token_id are None; cannot perform padding."
            )

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for idx, (ids, attn, lbl) in enumerate(
            zip(input_id_tensors, attention_tensors, label_tensors)
        ):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        # Process images
        pixel_tensors = []
        for sample in features:
            pv = sample["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        image_grid_thw = torch.stack(
            [
                torch.as_tensor(sample["image_grid_thw"], dtype=torch.long).view(-1)
                for sample in features
            ],
            dim=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# ==================== Main function ====================
def main():
    # ==================== Config ====================
    model_id = "./models/Qwen3-VL-8B-Instruct"  # change to your model path
    # model_id = "<BASE_PATH>/Qwen3-VL-8B-Merged-grounding"  # change to your model path
    # model_id = "<BASE_PATH>/Qwen3.5-9B"  # change to your model path
    output_dir = "./models/lora_checkpoints"  # change to your output path
    # train_data_path = "<BASE_PATH>/data/data_json.jsonl"  # change to your data path
    train_data_path = "./data/data.jsonl"  # change to your data path
    # train_data_path = "<BASE_PATH>/data/data_json_only_grounding.jsonl"  # change to your data path
    eval_data_path = "./data/eval_data_json.jsonl"  # optional

    MAX_LENGTH = 7000  # adjust based on VRAM
    BATCH_SIZE = 1  # for a single 4090 24G, recommend 1-2
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4

    # LoRA config
    LORA_RANK = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.01

    # ==================== Load model and processor ====================
    print("Loading model and processor...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="./cache",
        use_fast=False,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="./cache",
        use_fast=False,
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir="./cache",
        trust_remote_code=True,
    )

    # Enable Flash Attention 2
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"  # key: enable Flash Attention

    arch = (config.architectures or [None])[0]
    module_name = (
        f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    )
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)

    model = model_cls.from_pretrained(
        model_id,
        cache_dir="./cache",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 4090 supports bf16
        attn_implementation="flash_attention_2",  # key: enable Flash Attention
    )

    model.config.use_cache = False
    print(f"Model loaded, dtype: {model.dtype}")

    # ==================== Load dataset ====================
    print("Loading training dataset...")
    train_dataset = ForgeryAnalysisDataset(
        data_path=train_data_path,
        processor=processor,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    eval_dataset = None
    if os.path.exists(eval_data_path):
        print("Loading validation dataset...")
        eval_dataset = ForgeryAnalysisDataset(
            data_path=eval_data_path,
            processor=processor,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )

    """
    visual.patch_embed.proj - image patch embedding projection layer
    visual.merger.linear_fc1/linear_fc2 - feature merger
    visual.blocks.{0-26}.attn.qkv - attention QKV joint layer of each transformer block
    visual.blocks.{0-26}.attn.proj - attention output projection
    visual.blocks.{0-26}.mlp.linear_fc1/linear_fc2 - MLP layer
    visual.deepstack_merger_list.{0-2}.linear_fc1/linear_fc2 - deep stack merger
    """
    # layers = []
    # # blocks 0-26
    # for i in range(27):  # 0-26, 27 layers total
    #     # qkv layer
    #     layers.append(f"visual.blocks.{i}.attn.qkv")
    #     # proj layer
    #     layers.append(f"visual.blocks.{i}.attn.proj")
    # ==================== LoRA config ====================
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    # target_modules = target_modules + layers

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="lora_only",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.enable_input_require_grads()
    peft_model.print_trainable_parameters()

    # ==================== Training arguments ====================
    args = TrainingArguments(
        output_dir=output_dir,
        ddp_find_unused_parameters=False,  #
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=10,
        logging_first_step=True,
        num_train_epochs=NUM_EPOCHS,
        save_steps=500,
        save_total_limit=3,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,  # 4090 supports bf16
        fp16=False,
        report_to="tensorboard",  # key: use TensorBoard
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=12,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    # ==================== Create Trainer ====================
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenVLDataCollator(tokenizer=tokenizer),
    )

    # ==================== Start training ====================
    print("Starting training...")
    trainer.train()

    # # ==================== Save model ====================
    # print("Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    # under the subdirectory
    sub_dir = os.path.join(output_dir, "checkpoint-final")
    os.makedirs(sub_dir, exist_ok=True)
    trainer.model.save_pretrained(sub_dir)
    tokenizer.save_pretrained(sub_dir)
    processor.save_pretrained(sub_dir)

    # ==================== Plot the training loss curve ====================
    logs = trainer.state.log_history
    steps = [log["step"] for log in logs if "loss" in log]
    losses = [log["loss"] for log in logs if "loss" in log]

    if steps and losses:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        plt.close()
        print(
            f"Training loss curve saved to: {os.path.join(output_dir, 'training_loss.png')}"
        )

    print("Training complete!")


if __name__ == "__main__":
    main()
