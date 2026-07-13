from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_explainer.py"


def _remap_args(argv: list[str]) -> list[str]:
    alias_map = {
        "--model_name_or_path": "--model-name-or-path",
        "--train_jsonl": "--train-jsonl",
        "--val_jsonl": "--val-jsonl",
        "--output_dir": "--output-dir",
        "--max_length": "--max-length",
        "--train_batch_size": "--train-batch-size",
        "--eval_batch_size": "--eval-batch-size",
        "--grad_accum": "--grad-accum",
        "--learning_rate": "--learning-rate",
        "--warmup_ratio": "--warmup-ratio",
        "--weight_decay": "--weight-decay",
        "--max_grad_norm": "--max-grad-norm",
        "--logging_steps": "--logging-steps",
        "--save_steps": "--save-steps",
        "--eval_steps": "--eval-steps",
        "--save_total_limit": "--save-total-limit",
        "--gradient_checkpointing": "--gradient-checkpointing",
        "--load_in_4bit": "--load-in-4bit",
        "--load_in_8bit": "--load-in-8bit",
        "--lora_r": "--lora-r",
        "--lora_alpha": "--lora-alpha",
        "--lora_dropout": "--lora-dropout",
        "--lora_target_modules": "--lora-target-modules",
        "--max_train_samples": "--max-train-samples",
        "--max_val_samples": "--max-val-samples",
        "--resume_from_checkpoint": "--resume-from-checkpoint",
    }
    remapped: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            name, value = arg.split("=", 1)
            normalized = alias_map.get(name, name.replace("_", "-"))
            remapped.append(f"{normalized}={value}")
        elif arg.startswith("--"):
            remapped.append(alias_map.get(arg, arg.replace("_", "-")))
        else:
            remapped.append(arg)
    return remapped


def main() -> None:
    sys.argv = [str(SCRIPT_PATH), *_remap_args(sys.argv[1:])]
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
