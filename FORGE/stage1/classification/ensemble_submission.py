import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble fold probability CSVs by averaging probabilities, then threshold for final labels."
    )
    parser.add_argument(
        "--prob-csvs",
        nargs="+",
        required=True,
        help="List of fold probability CSVs (each must contain columns: Image_name, prob).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./submission_ensemble.csv",
        help="Output CSV path for ensembled labels.",
    )
    parser.add_argument(
        "--template-csv",
        type=str,
        default="",
        help="Optional template submission CSV containing Image_name. If provided, output follows template order.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method across fold probabilities.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold if no threshold file(s) are provided.",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="",
        help="Optional single threshold metadata JSON with key 'threshold'.",
    )
    parser.add_argument(
        "--threshold-files",
        nargs="+",
        default=None,
        help="Optional multiple threshold metadata JSON files. Their thresholds are averaged.",
    )
    return parser.parse_args()


def _load_threshold_from_json(path: str) -> float:
    json_path = os.path.abspath(path)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Threshold metadata file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError(
            f"{json_path} must be a JSON object containing key 'threshold'."
        )

    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold in {json_path} must be in [0.0, 1.0], got {threshold}"
        )
    return threshold


def resolve_threshold(args: argparse.Namespace) -> Tuple[float, str]:
    if args.threshold_file and args.threshold_files:
        raise ValueError("Use only one of --threshold-file or --threshold-files.")

    if args.threshold_files:
        thresholds = [_load_threshold_from_json(path) for path in args.threshold_files]
        if len(thresholds) == 0:
            raise ValueError("--threshold-files is empty.")
        threshold = float(np.mean(np.asarray(thresholds, dtype=np.float32)))
        return threshold, f"mean_of_{len(thresholds)}_threshold_files"

    if args.threshold_file:
        threshold = _load_threshold_from_json(args.threshold_file)
        return threshold, "threshold_file"

    threshold = float(args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0.0, 1.0], got {threshold}")
    return threshold, "fixed_threshold"


def _read_prob_csv(path: str) -> pd.DataFrame:
    csv_path = os.path.abspath(path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Probability CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Image_name", "prob"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {missing_cols}")

    out = df[["Image_name", "prob"]].copy()
    out["Image_name"] = out["Image_name"].astype(str)
    out["prob"] = pd.to_numeric(out["prob"], errors="coerce")

    if out["Image_name"].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicated Image_name entries.")
    if out["prob"].isna().any():
        raise ValueError(f"{csv_path} contains non-numeric or missing prob values.")

    return out


def aggregate_probabilities(
    prob_csv_paths: List[str], agg: str
) -> Tuple[List[str], np.ndarray]:
    if len(prob_csv_paths) == 0:
        raise ValueError("No probability CSV provided.")

    first_df = _read_prob_csv(prob_csv_paths[0])
    image_names = first_df["Image_name"].tolist()
    first_name_set = set(image_names)

    prob_arrays = [first_df["prob"].to_numpy(dtype=np.float32)]

    for path in prob_csv_paths[1:]:
        df = _read_prob_csv(path)
        name_set = set(df["Image_name"].tolist())

        missing = first_name_set - name_set
        extra = name_set - first_name_set
        if missing or extra:
            raise ValueError(
                f"Image_name mismatch between probability CSVs. "
                f"Missing={len(missing)} Extra={len(extra)} in {os.path.abspath(path)}"
            )

        aligned = df.set_index("Image_name").reindex(image_names)
        if aligned["prob"].isna().any():
            raise ValueError(
                f"Failed to align probabilities by Image_name for {os.path.abspath(path)}"
            )

        prob_arrays.append(aligned["prob"].to_numpy(dtype=np.float32))

    stacked = np.stack(prob_arrays, axis=0)
    if agg == "mean":
        aggregated = np.mean(stacked, axis=0)
    else:
        aggregated = np.median(stacked, axis=0)

    return image_names, aggregated


def build_submission_dataframe(
    template_csv: str, image_names: List[str], labels: np.ndarray
) -> pd.DataFrame:
    label_list = labels.astype(np.int32).tolist()

    if template_csv:
        template_path = os.path.abspath(template_csv)
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template CSV not found: {template_path}")

        template_df = pd.read_csv(template_path)
        if "Image_name" not in template_df.columns:
            raise ValueError(
                f"Template CSV missing required column: Image_name ({template_path})"
            )

        mapping = {name: int(label) for name, label in zip(image_names, label_list)}
        template_df["label"] = (
            template_df["Image_name"].map(mapping).fillna(0).astype(int)
        )
        return template_df

    return pd.DataFrame({"Image_name": image_names, "label": label_list})


def main() -> None:
    args = parse_args()

    prob_csv_paths = [os.path.abspath(path) for path in args.prob_csvs]
    image_names, aggregated_probs = aggregate_probabilities(
        prob_csv_paths, agg=args.agg
    )

    threshold, threshold_source = resolve_threshold(args)
    labels = (aggregated_probs > threshold).astype(np.int32)

    output_df = build_submission_dataframe(args.template_csv, image_names, labels)

    output_csv = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    prob_csv = os.path.splitext(output_csv)[0] + "_prob.csv"
    prob_df = pd.DataFrame(
        {
            "Image_name": image_names,
            "prob": aggregated_probs.astype(np.float32),
            "label": labels.astype(np.int32),
        }
    )
    prob_df.to_csv(prob_csv, index=False)

    print(f"Loaded fold probability CSVs: {len(prob_csv_paths)}")
    print(f"Aggregation method: {args.agg}")
    print(f"Threshold source: {threshold_source}")
    print(f"Classification threshold used: {threshold:.4f}")
    print(f"Saved ensembled label CSV: {output_csv}")
    print(f"Saved ensembled probability CSV: {prob_csv}")


if __name__ == "__main__":
    main()
