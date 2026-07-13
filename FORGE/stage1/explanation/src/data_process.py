from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_sft_data.py"


def _remap_args(argv: list[str]) -> list[str]:
    alias_map = {
        "--data_root": "--data-root",
        "--output_dir": "--output-dir",
        "--val_ratio": "--val-ratio",
        "--max_black": "--max-black",
        "--max_white": "--max-white",
        "--enable_vl": "--enable-vl",
        "--vl_model_name_or_path": "--vl-model-name-or-path",
        "--vl_load_in_4bit": "--vl-load-in-4bit",
        "--vl_load_in_8bit": "--vl-load-in-8bit",
        "--vl_torch_dtype": "--vl-torch-dtype",
        "--vl_batch_size": "--vl-batch-size",
        "--vl_trust_remote_code": "--vl-trust-remote-code",
        "--vl_max_regions": "--vl-max-regions",
        "--vl_context_ratio": "--vl-context-ratio",
        "--vl_min_crop_size": "--vl-min-crop-size",
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
