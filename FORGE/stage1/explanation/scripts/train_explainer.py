from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

    _TORCH_IMPORT_ERROR = exc

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _TRANSFORMERS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    Trainer = None
    TrainingArguments = None
    set_seed = None
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _PEFT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    _PEFT_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.explainer.runtime_env import sanitize_thread_env

sanitize_thread_env()

from src.explainer.prompting import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning: explainable image-forgery-detection text model"
    )

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "sft_train.jsonl"),
        help="Training set JSONL path",
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "sft_val.jsonl"),
        help="Validation set JSONL path (may not exist)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "explainer_qlora"),
        help="Training output directory",
    )

    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum token length"
    )
    parser.add_argument(
        "--epochs", type=float, default=3.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=1, help="Per-GPU training batch size"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=1, help="Per-GPU validation batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Gradient clipping"
    )

    parser.add_argument(
        "--logging-steps", type=int, default=20, help="Logging interval"
    )
    parser.add_argument(
        "--save-steps", type=int, default=200, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=200, help="Validation interval"
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load in 4-bit QLoRA",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit",
    )
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load in 8-bit (mutually exclusive with 4-bit)",
    )

    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="LoRA target modules (comma-separated)",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=-1,
        help="Maximum number of training samples to use, -1 for all",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=-1,
        help="Maximum number of validation samples to use, -1 for all",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Checkpoint path to resume training from",
    )

    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int = -1) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_chat_template(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    instruction: str,
    add_generation_prompt: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    base = f"system: {system_prompt}\nuser: {instruction}\n"
    if add_generation_prompt:
        base += "assistant: "
    return base


class ExplanationSFTDataset(Dataset):
    def __init__(
        self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int
    ):
        self.samples: List[Dict[str, List[int]]] = []
        dropped = 0

        eos_text = tokenizer.eos_token or ""

        for record in records:
            instruction = str(record.get("instruction", "")).strip()
            target = str(record.get("target", "")).strip()
            if not instruction or not target:
                dropped += 1
                continue

            prompt_text = apply_chat_template(
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT,
                instruction=instruction,
                add_generation_prompt=True,
            )
            full_text = prompt_text + target + eos_text

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            if len(full_ids) == 0:
                dropped += 1
                continue

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            start = min(len(prompt_ids), len(full_ids) - 1)
            labels = [-100] * len(full_ids)
            for idx in range(start, len(full_ids)):
                labels[idx] = full_ids[idx]

            if all(value == -100 for value in labels):
                dropped += 1
                continue

            self.samples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

        self.dropped = dropped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]


class SFTDataCollator:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


def build_training_arguments(training_kwargs: Dict[str, Any]):
    """Accommodate parameter naming differences for TrainingArguments across transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    if (
        "evaluation_strategy" in training_kwargs
        and "evaluation_strategy" not in params
        and "eval_strategy" in params
    ):
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    elif (
        "eval_strategy" in training_kwargs
        and "eval_strategy" not in params
        and "evaluation_strategy" in params
    ):
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")

    return TrainingArguments(**training_kwargs)


def build_model_and_tokenizer(args: argparse.Namespace):
    if (
        AutoTokenizer is None
        or AutoModelForCausalLM is None
        or BitsAndBytesConfig is None
    ):
        raise ImportError(
            "transformers is missing in the current environment; training cannot proceed. Please run `pip install -r requirements.txt` first."
        ) from _TRANSFORMERS_IMPORT_ERROR

    if torch is None:
        raise ImportError(
            "torch is missing in the current environment; training cannot proceed. Please run `pip install -r requirements.txt` first."
        ) from _TORCH_IMPORT_ERROR

    if (
        LoraConfig is None
        or get_peft_model is None
        or prepare_model_for_kbit_training is None
    ):
        raise ImportError(
            "peft is missing in the current environment; LoRA training cannot proceed. Please run `pip install -r requirements.txt` first."
        ) from _PEFT_IMPORT_ERROR

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("4-bit and 8-bit cannot be enabled at the same time")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=False, use_fast=False
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "trust_remote_code": False,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = [
        name.strip() for name in args.lora_target_modules.split(",") if name.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    model.print_trainable_parameters()
    return tokenizer, model


def main() -> None:
    args = parse_args()

    if torch is None:
        raise ImportError(
            "torch is missing in the current environment; cannot run the training script. Please install requirements.txt first."
        ) from _TORCH_IMPORT_ERROR
    if set_seed is None or Trainer is None or TrainingArguments is None:
        raise ImportError(
            "transformers is missing in the current environment; cannot run the training script. Please install requirements.txt first."
        ) from _TRANSFORMERS_IMPORT_ERROR

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(
        Path(args.train_jsonl), max_samples=args.max_train_samples
    )
    val_records = load_jsonl(Path(args.val_jsonl), max_samples=args.max_val_samples)

    if not train_records:
        raise RuntimeError(f"Training set is empty: {args.train_jsonl}")

    tokenizer, model = build_model_and_tokenizer(args)

    train_dataset = ExplanationSFTDataset(
        train_records, tokenizer=tokenizer, max_length=args.max_length
    )
    val_dataset = (
        ExplanationSFTDataset(
            val_records, tokenizer=tokenizer, max_length=args.max_length
        )
        if val_records
        else None
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training samples are empty after tokenization; please check the input data"
        )

    print(f"Train samples: {len(train_dataset)}, dropped: {train_dataset.dropped}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}, dropped: {val_dataset.dropped}")

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        raise RuntimeError(
            "tokenizer is missing pad/eos token id; cannot perform batch padding"
        )

    data_collator = SFTDataCollator(pad_token_id=pad_token_id)
    has_eval = val_dataset is not None and len(val_dataset) > 0

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "fp16": not args.bf16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "evaluation_strategy": "steps" if has_eval else "no",
        "save_strategy": "steps",
    }

    if has_eval:
        training_kwargs.update(
            {
                "eval_steps": args.eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )

    training_args = build_training_arguments(training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_eval else None,
        data_collator=data_collator,
    )

    resume_path = (
        args.resume_from_checkpoint.strip() if args.resume_from_checkpoint else None
    )
    train_result = trainer.train(resume_from_checkpoint=resume_path)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_history = list(trainer.state.log_history)
    save_json(output_dir / "train_log_history.json", log_history)
    save_jsonl(
        output_dir / "train_log_history.jsonl", [dict(item) for item in log_history]
    )

    train_result_metrics = (
        dict(train_result.metrics) if train_result is not None else {}
    )
    save_json(
        output_dir / "train_result_metrics.json",
        {
            "global_step": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "metrics": train_result_metrics,
        },
    )

    meta = {
        "base_model": args.model_name_or_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset) if val_dataset is not None else 0,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_result_metrics": train_result_metrics,
        "log_history_entries": len(log_history),
        "args": vars(args),
    }
    save_json(output_dir / "train_meta.json", meta)

    print(f"Training complete; LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
