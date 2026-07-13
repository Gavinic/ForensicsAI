"""
Post-processing script for balancing label distribution.

Logic:
- If the ratio of label=0 to label=1 is not 1:4, and label=1 has more samples,
  then flip the 3 lowest confidence label=1 samples to label=0.
"""

import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process submission CSV to balance label distribution."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input submission CSV (with Image_name and label columns).",
    )
    parser.add_argument(
        "--prob-csv",
        required=True,
        help="Path to probability CSV (with Image_name, prob, and label columns).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path to output CSV. Default: overwrite input CSV.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.25,
        help="Target ratio of label=0 to label=1 (default: 0.25 = 1:4).",
    )
    parser.add_argument(
        "--num-flip",
        type=int,
        default=3,
        help="Number of lowest confidence label=1 samples to flip (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = os.path.abspath(args.input_csv)
    prob_csv = os.path.abspath(args.prob_csv)
    output_csv = os.path.abspath(args.output_csv) if args.output_csv else input_csv

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not os.path.isfile(prob_csv):
        raise FileNotFoundError(f"Probability CSV not found: {prob_csv}")

    df = pd.read_csv(input_csv)
    prob_df = pd.read_csv(prob_csv)

    if "label" not in df.columns:
        raise ValueError("Input CSV must have 'label' column.")
    if "Image_name" not in df.columns:
        raise ValueError("Input CSV must have 'Image_name' column.")
    if "prob" not in prob_df.columns:
        raise ValueError("Probability CSV must have 'prob' column.")
    if "Image_name" not in prob_df.columns:
        raise ValueError("Probability CSV must have 'Image_name' column.")

    count_0 = int((df["label"] == 0).sum())
    count_1 = int((df["label"] == 1).sum())
    total = count_0 + count_1

    print(f"Original distribution: label=0: {count_0}, label=1: {count_1}")
    print(
        f"Ratio label=0 / total: {count_0 / total:.4f} (target: {args.target_ratio:.4f})"
    )

    current_ratio = count_0 / total if total > 0 else 0

    if count_1 > count_0 and abs(current_ratio - args.target_ratio) > 1e-6:
        print(
            f"Ratio not satisfied (current: {current_ratio:.4f}, target: {args.target_ratio:.4f})"
        )
        print(
            f"label=1 has more samples, flipping {args.num_flip} lowest confidence label=1 samples..."
        )

        label1_df = prob_df[prob_df["label"] == 1].copy()
        label1_df = label1_df.sort_values(by="prob", ascending=True)

        num_to_flip = min(args.num_flip, len(label1_df))
        flip_names = label1_df.head(num_to_flip)["Image_name"].tolist()

        flip_mask = df["Image_name"].isin(flip_names) & (df["label"] == 1)
        df.loc[flip_mask, "label"] = 0

        new_count_0 = int((df["label"] == 0).sum())
        new_count_1 = int((df["label"] == 1).sum())
        print(f"Flipped {num_to_flip} samples.")
        print(f"New distribution: label=0: {new_count_0}, label=1: {new_count_1}")
    else:
        print("Ratio already satisfied or label=0 has more samples. No changes needed.")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
