from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_explanations.py"


def _remap_args(argv: list[str]) -> list[str]:
    alias_map = {
        "--input_path": "--answer-csv",
        "--output_path": "--output-csv",
        "--base_model_dir": "--model-name-or-path",
        "--vl_model_dir": "--vl-model-name-or-path",
        "--test_image_dir": "--test-image-dir",
        "--adapter_path": "--adapter-path",
        "--disable_llm": "--disable-llm",
        "--load_in_4bit": "--load-in-4bit",
        "--load_in_8bit": "--load-in-8bit",
        "--torch_dtype": "--torch-dtype",
        "--enable_vl": "--enable-vl",
        "--vl_load_in_4bit": "--vl-load-in-4bit",
        "--vl_load_in_8bit": "--vl-load-in-8bit",
        "--vl_torch_dtype": "--vl-torch-dtype",
        "--vl_batch_size": "--vl-batch-size",
        "--vl_trust_remote_code": "--vl-trust-remote-code",
        "--vl_max_regions": "--vl-max-regions",
        "--vl_context_ratio": "--vl-context-ratio",
        "--vl_min_crop_size": "--vl-min-crop-size",
        "--max_new_tokens": "--max-new-tokens",
        "--top_p": "--top-p",
        "--repetition_penalty": "--repetition-penalty",
        "--gen_batch_size": "--gen-batch-size",
        "--grounded_rewrite": "--grounded-rewrite",
        "--preprocess_workers": "--preprocess-workers",
        "--max_samples": "--max-samples",
        "--evidence_jsonl": "--evidence-jsonl",
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
