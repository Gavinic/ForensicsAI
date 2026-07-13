import json
import os
import re

import pandas as pd
from PIL import Image

# ================= Config =================
# Dataset root directory (contains Black and White folders)
DATASET_ROOT = "./data/ForgeryAnalysis_Stage_1_Train"
# Output training data file
OUTPUT_JSONL = "./data/data_json.jsonl"


def process_dataset():
    # Regex matching coordinates, e.g. [364, 932, 388, 957]
    coord_pattern = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
    # List of bad images to skip
    bad_images = [
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/48d65ccd195745b2893feb5d62b6de34.png",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/6f35c29554104413ba5cc10b37c0c971.png",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/72bf5c64706245f7b5586f0b482ada8a.png",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/a394af3681bd49349f25b4884ba3f791.jpg",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/c08af85a286946cea8117d6cd8dbc5b9.png",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/ca10ecb94c624b0ebedf8b600977fd6a.png",
        "./data/ForgeryAnalysis_Stage_1_Train/Black/Image/ed757e5a047c41ec84971aee79c61153.jpg",
    ]
    # Convert to a set for fast lookup
    bad_images_set = set(bad_images)

    dataset_records1 = []
    dataset_records2 = []
    categories = {"Black": 1, "White": 0}

    for category, label in categories.items():
        category_dir = os.path.join(DATASET_ROOT, category)
        caption_dir = os.path.join(category_dir, "Caption")
        # Assume images are stored in the sibling image folder as jpg. Modify as needed.
        image_dir = os.path.join(category_dir, "Image")

        if not os.path.exists(caption_dir):
            continue

        for md_filename in os.listdir(caption_dir):
            if not md_filename.endswith(".md"):
                continue

            base_name = os.path.splitext(md_filename)[0]
            md_path = os.path.join(caption_dir, md_filename)

            # Find the corresponding image (supports common format extensions)
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                temp_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break

            if not img_path:
                print(
                    f"Warning: could not find the image file for {md_filename}, skipping this sample."
                )
                continue

            # Check whether it is a bad image; if so, skip
            if img_path in bad_images_set:
                print(f"Skipping bad image: {img_path}")
                continue

            # Get image width and height
            try:
                with Image.open(img_path) as img:
                    W, H = img.size
            except Exception as e:
                print(f"Failed to read image {img_path}: {e}")
                continue

            # Read MD content
            with open(md_path, "r", encoding="utf-8") as f:
                original_text = f.read().strip()

            boxes_list = []

            # Define a replacement function used during regex matching
            def replace_and_normalize(match):
                x1, y1, x2, y2 = map(int, match.groups())

                # Normalize to 0-1000 and ensure no out-of-bounds
                nx1 = max(0, min(1000, int((x1 / W) * 1000)))
                ny1 = max(0, min(1000, int((y1 / H) * 1000)))
                nx2 = max(0, min(1000, int((x2 / W) * 1000)))
                ny2 = max(0, min(1000, int((y2 / H) * 1000)))

                # Append to the separate boxes list
                boxes_list.append([nx1, ny1, nx2, ny2])

                # Return the replaced string, put back in place
                return f"[{nx1}, {ny1}, {nx2}, {ny2}]"

            # Use sub to replace all coordinates in the text while collecting boxes
            normalized_text = coord_pattern.sub(replace_and_normalize, original_text)

            # Build the expected JSON dict that the model should output
            assistant_output_dict = {
                "boxes": boxes_list,
                "label": label,
                "explanation": normalized_text,
            }
            # Convert dict to string as the large model's prediction target
            assistant_output_str = json.dumps(assistant_output_dict, ensure_ascii=False)
            record1 = {
                "image": img_path,
                "explanation": assistant_output_str,
            }
            record2 = {
                "image": img_path,
                "explanation": str(boxes_list),
                "grounding": 1,
            }

            dataset_records1.append(record1)
            dataset_records2.append(record2)
    # Write JSONL file
    result = dataset_records1 + dataset_records2
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in result:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Data processing complete! Generated {len(result)} training records, saved to {OUTPUT_JSONL}."
    )


if __name__ == "__main__":
    process_dataset()
