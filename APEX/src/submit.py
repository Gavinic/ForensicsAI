import argparse
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean text and remove think tags"""
    # remove <think>...</think> and its contents
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # existing cleaning logic
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    # create argument parser
    parser = argparse.ArgumentParser(description="Output path")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output directory"
    )

    args = parser.parse_args()
    # get paths
    root_dir = Path(__file__).parent.parent.absolute()
    output_dir = os.path.join(root_dir, args.output_path)

    csv_path = os.path.join(root_dir, "tmp_dir/tmp.csv")
    cap_dir = os.path.join(root_dir, "tmp_dir/Output_Caption")
    df = pd.read_csv(csv_path)

    explanations = []
    for idx, row in tqdm(df.iterrows()):
        img_id = row["image_name"].split(".")[0]
        cp_path = os.path.join(cap_dir, f"{img_id}.md")
        with open(cp_path, "r", encoding="utf-8") as f:
            description = f.read()
            explanations.append(clean_text(description))

    df["explanation"] = explanations
    df.to_csv(os.path.join(output_dir, "submit.csv"), index=False)


if __name__ == "__main__":
    main()
