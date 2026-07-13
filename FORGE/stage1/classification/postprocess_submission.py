import argparse
import json
import os
import re
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process submission CSV: for label=0, set location.counts to empty string."
    )
    parser.add_argument(
        "--input-csv", required=True, help="Path to input submission CSV."
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path to output CSV. Default: <input>_label0_emptycounts.csv",
    )
    return parser.parse_args()


def _empty_counts(location: Any) -> str:
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return json.dumps({"size": [0, 0], "counts": ""}, separators=(",", ":"))

    text = str(location).strip()
    if not text:
        return json.dumps({"size": [0, 0], "counts": ""}, separators=(",", ":"))

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload["counts"] = ""
            if "size" not in payload:
                payload["size"] = [0, 0]
            return json.dumps(payload, separators=(",", ":"))
    except json.JSONDecodeError:
        pass

    # Fallback for non-standard strings: replace only the first counts value.
    replaced = re.sub(r'("counts"\s*:\s*")[^"]*(")', r"\1\2", text, count=1)
    return replaced


def main() -> None:
    args = parse_args()

    input_csv = os.path.abspath(args.input_csv)
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if args.output_csv:
        output_csv = os.path.abspath(args.output_csv)
    else:
        stem, ext = os.path.splitext(input_csv)
        output_csv = f"{stem}_label0_emptycounts{ext or '.csv'}"

    df = pd.read_csv(input_csv)

    for required_col in ("label", "location"):
        if required_col not in df.columns:
            raise ValueError(f"Missing required column: {required_col}")

    label_num = pd.to_numeric(df["label"], errors="coerce")
    label0_mask = label_num.eq(0)
    updated_count = int(label0_mask.sum())

    df.loc[label0_mask, "location"] = df.loc[label0_mask, "location"].apply(
        _empty_counts
    )

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} rows. Updated {updated_count} label=0 rows.")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
