import argparse
import os
import subprocess
import sys


def run_script(script_name, args_list):
    """Generic function: run a Python script and pass through arguments."""
    cmd = [sys.executable, script_name] + args_list
    print(f"\n>>> Running: {' '.join(cmd)}")

    # subprocess.run waits for the script to finish before continuing
    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        print(f"Error: {script_name} failed.")
        sys.exit(1)
    print(f">>> {script_name} finished.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Forgery Analysis inference integration script (Stage 2)"
    )

    # --- Common arguments ---
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="<BASE_PATH>/models/qwen3-vl-8b-instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="../../datasets/ForgeryAnalysis_Stage_2_Test/Image",
        help="Test image directory",
    )
    parser.add_argument(
        "--lora_a_model_path",
        type=str,
        default="../../models/adapters/stage2_final/lora_a/",
        help="LoRA A model path",
    )
    parser.add_argument(
        "--lora_b_model_path",
        type=str,
        default="../../models/adapters/stage2_final/lora_b/",
        help="LoRA B model path",
    )
    parser.add_argument(
        "--segformer_model_path",
        type=str,
        default="../../models/adapters/stage2_final/segformer.pth",
        help="SegFormer model path",
    )

    parser.add_argument(
        "--test_ocr_path",
        type=str,
        default="../../datasets/stage2_final/ocr_test.xlsx",
        help="OCR test file path",
    )

    parser.add_argument(
        "--mask_dir", type=str, default="output_masks", help="Mask directory"
    )
    parser.add_argument(
        "--output_path", type=str, default="result.csv", help="Final result save path"
    )

    parser.add_argument(
        "--best",
        action="store_true",
        help="Produce the highest-scoring inference result",
    )
    parser.add_argument(
        "--best_expn_model_path",
        type=str,
        default="../../models/adapters/stage2_final/expn_over/",
        help="Best explanation model path",
    )

    args = parser.parse_args()

    # --- Run the first script: Classification & Explanation ---
    script1_args = [
        "--test_ocr_path",
        args.test_ocr_path,
        "--input_path",
        args.input_path,
        "--base_model_path",
        args.base_model_path,
        "--lora_a_model_path",
        args.lora_a_model_path,
        "--best_expn_model_path",
        args.best_expn_model_path,
    ]
    if args.best:
        script1_args.append("--best")
    run_script("code_infer_11_classification_v4_score_explanation.py", script1_args)

    # --- Run the second script: Grounding ---
    script2_args = [
        "--base_model_path",
        args.base_model_path,
        "--lora_b_model_path",
        args.lora_b_model_path,
        "--input_path",
        args.input_path,
        "--best_expn_model_path",
        args.best_expn_model_path,
    ]
    if args.best:
        script2_args.append("--best")
    run_script("code_infer_08_grounding.py", script2_args)

    # --- Run the third script: Traditional Grounding ---
    script3_args = [
        "--input_path",
        args.mask_dir,
        "--segformer_model_path",
        args.segformer_model_path,
    ]
    if args.best:
        script3_args.append("--best")
    run_script("code_infer_09_traditional_grounding.py", script3_args)

    # --- Run the fourth script: Get Final Result ---
    script4_args = [
        "--mask_dir",
        args.mask_dir,
        "--output_path",
        args.output_path,
    ]
    if args.best:
        script4_args.append("--best")
    run_script("code_infer_100_result.py", script4_args)

    print("=" * 40)
    print("All inference and result generation tasks completed successfully!")
    print(f"Final result saved to: {args.output_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
