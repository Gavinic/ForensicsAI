from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.explainer.runtime_env import sanitize_thread_env

sanitize_thread_env()

from src.explainer.dataset_builder import prepare_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SFT dataset for image-forgery-detection explanations"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory; will generate sft_train.jsonl / sft_val.jsonl",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation set ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-black",
        type=int,
        default=-1,
        help="Maximum number of Black Caption samples to use; -1 for all",
    )
    parser.add_argument(
        "--max-white",
        type=int,
        default=-1,
        help="Maximum number of White Caption samples to use; -1 for all",
    )
    parser.add_argument(
        "--enable-vl",
        action="store_true",
        help="Call a VL model during data construction to generate global/local observations",
    )
    parser.add_argument(
        "--vl-model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VL model name or local path",
    )
    parser.add_argument(
        "--vl-load-in-4bit",
        dest="vl_load_in_4bit",
        action="store_true",
        help="Load VL model in 4-bit",
    )
    parser.add_argument(
        "--no-vl-load-in-4bit",
        dest="vl_load_in_4bit",
        action="store_false",
        help="Disable VL model 4-bit",
    )
    parser.set_defaults(vl_load_in_4bit=True)
    parser.add_argument(
        "--vl-load-in-8bit",
        action="store_true",
        help="Load VL model in 8-bit (mutually exclusive with 4-bit)",
    )
    parser.add_argument(
        "--vl-torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bf16", "bfloat16"],
    )
    parser.add_argument(
        "--vl-batch-size",
        type=int,
        default=4,
        help="VL batch inference size, used to improve GPU utilization",
    )
    parser.add_argument(
        "--vl-trust-remote-code",
        action="store_true",
        help="Allow the VL model to load remote custom code",
    )
    parser.add_argument(
        "--vl-max-regions",
        type=int,
        default=3,
        help="Maximum number of suspicious regions to observe per image",
    )
    parser.add_argument(
        "--vl-context-ratio",
        type=float,
        default=0.35,
        help="Context expansion ratio when cropping local regions",
    )
    parser.add_argument(
        "--vl-min-crop-size",
        type=int,
        default=224,
        help="Minimum side length for local crops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_sft_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_black_records=args.max_black,
        max_white_records=args.max_white,
        enable_vl=args.enable_vl,
        vl_model_name_or_path=args.vl_model_name_or_path,
        vl_load_in_4bit=args.vl_load_in_4bit,
        vl_load_in_8bit=args.vl_load_in_8bit,
        vl_torch_dtype=args.vl_torch_dtype,
        vl_trust_remote_code=args.vl_trust_remote_code,
        vl_batch_size=args.vl_batch_size,
        vl_max_regions=args.vl_max_regions,
        vl_context_ratio=args.vl_context_ratio,
        vl_min_crop_size=args.vl_min_crop_size,
    )

    print("=== SFT data construction complete ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
