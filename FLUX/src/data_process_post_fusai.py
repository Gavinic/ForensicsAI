import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

# ================= Config =================
# Input file path (assume you saved the provided sample data as predictions.json)
INPUT_JSON_PATH = "./data/result/result_fusai.json"
# Image data root directory (for reading image dimensions)
IMAGE_ROOT_DIR = "./data/ForgeryAnalysis_Stage_2_Test/Image/"
# Output CSV path
OUTPUT_CSV_PATH = "./data/result/result_fusai.csv"
# ===========================================


def load_data(path):
    """Load raw prediction JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_size(image_path):
    """Get image dimensions (Height, Width)."""
    try:
        # Support both absolute and relative paths
        if not os.path.exists(image_path):
            # Try joining with the root directory
            filename = os.path.basename(image_path)
            image_path = os.path.join(IMAGE_ROOT_DIR, filename)

        with Image.open(image_path) as img:
            return img.size[1], img.size[0]  # return H, W
    except Exception as e:
        print(
            f"Warning: Could not load image {image_path}, using default size 1024x1024. Error: {e}"
        )
        return 1024, 1024


def extract_boxes_from_text(text):
    """Extract the coordinate list [] from the explanation text."""
    # Match structures like [100, 200, 300, 400] or [0.1, 0.2, 0.3, 0.4]
    pattern = r"\[(\d+\.?\d*(?:,\s*\d+\.?\d*)*)\]"
    matches = re.findall(pattern, text)
    boxes = []
    for m in matches:
        nums = [float(x.strip()) for x in m.split(",")]
        # Only keep data that looks like a coordinate box (at least 4 numbers)
        if len(nums) >= 4:
            # Take the first 4 as x1, y1, x2, y2
            boxes.append(nums[:4])
    return boxes


def infer_label(explanation, json_label=None):
    """Infer the label: prefer the JSON field."""
    # if len(explanation) == 0:
    #     print("Warning: JSON label missing")
    #     return 1
    # else:
    #     if "This is a real" in explanation or "This is a real" in explanation:
    #         return 0
    #     else:
    #         return 1
    if json_label is not None:
        return int(json_label)
    else:
        print("Warning: JSON label missing")
        return 1


# ================= Modification starts =================
def update_explanation_and_get_boxes(text, img_h, img_w):
    """
    Extract coordinate boxes from the text, denormalize them, and write the
    denormalized coordinates back into the original text.
    Returns: (updated text, list of denormalized coordinates)
    """
    boxes = []
    # Match coordinate boxes of the form [x1, y1, x2, y2], supporting integers and floats, allowing spaces
    # Use capture groups to get the 4 coordinate values separately
    pattern = (
        r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]"
    )

    def repl(match):
        # Get the 4 captured coordinate values
        nums = [float(g) for g in match.groups()]

        # ================= Denormalization logic =================
        # Determine whether it is a normalized coordinate: if the max value <= 1000, treat as normalized (because it was multiplied by 1000)
        if max(nums) <= 1000:
            # Denormalization formula: coordinate value / 1000 * image dimension
            x1 = int((nums[0] / 1000.0) * img_w)
            y1 = int((nums[1] / 1000.0) * img_h)
            x2 = int((nums[2] / 1000.0) * img_w)
            y2 = int((nums[3] / 1000.0) * img_h)
        else:
            # Already pixel coordinates, use directly
            x1, y1, x2, y2 = [int(n) for n in nums]
        # ========================================================

        # Boundary protection
        x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
        y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))

        # Ensure x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Save to the list for RLE use
        boxes.append([x1, y1, x2, y2])

        # Return the formatted coordinate string, replacing the coordinates in the original text
        return f"[{x1}, {y1}, {x2}, {y2}]"

    # Perform the replacement
    new_text = re.sub(pattern, repl, text)
    return new_text, boxes


def denormalize_boxes(boxes, img_h, img_w):
    """Denormalize coordinates: if a value <= 1000 treat it as a normalized coordinate."""
    new_boxes = []
    for box in boxes:
        # Determine whether normalized (assume max value <= 1000 means normalized)
        if max(box) <= 1000:
            # Denormalization formula: coordinate value / 1000 * image dimension
            x1 = int((box[0] / 1000.0) * img_w)
            y1 = int((box[1] / 1000.0) * img_h)
            x2 = int((box[2] / 1000.0) * img_w)
            y2 = int((box[3] / 1000.0) * img_h)
        else:
            # Already pixel coordinates, use directly
            x1, y1, x2, y2 = [int(n) for n in box]
        # Boundary protection
        x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
        y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))

        # Ensure x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Filter out boxes that are too small
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            new_boxes.append([x1, y1, x2, y2])
    return new_boxes


def boxes_to_rle(boxes, img_h, img_w):
    """Convert boxes to a mask and encode it in COCO RLE format."""
    if not boxes:
        # When boxes is empty, return an empty RLE structure
        rle_dict = {
            "size": [int(img_h), int(img_w)],
            "counts": "",
        }
        return json.dumps(rle_dict)
    # Create a blank mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            mask[y1:y2, x1:x2] = 1

    # Use pycocotools for RLE encoding
    # Note: pycocotools requires the mask to be in Fortran order
    rle = mask_utils.encode(np.asfortranarray(mask))

    # Convert to the dict format required by the submission spec; counts must be decoded to a string
    # pycocotools outputs counts as bytes; convert to string for JSON serialization
    rle_dict = {
        "size": [int(img_h), int(img_w)],
        "counts": (
            rle["counts"].decode("utf-8")
            if isinstance(rle["counts"], bytes)
            else rle["counts"]
        ),
    }
    return json.dumps(rle_dict)


def process_data(data_list):
    """Main processing pipeline."""
    results = []

    for item in data_list:
        image_id = item.get("img_path", "")
        image_name = os.path.basename(image_id)
        raw_output_str = item.get("explanation", "{}")

        # 1. Parse raw_output
        try:
            # Handle possible nested JSON strings
            if isinstance(raw_output_str, str):
                if not raw_output_str.endswith('"}'):
                    raw_output_str += '"}'
                raw_data = json.loads(raw_output_str)
            else:
                raw_data = raw_output_str
        except json.JSONDecodeError:
            print(f"Error parsing JSON for {image_name}")
            raw_data = {}

        explanation = raw_data.get("explanation", "")
        json_label = raw_data.get("label", None)
        json_boxes = raw_data.get("boxes", None)

        # 2. Determine the label
        label = infer_label(explanation, json_label)

        # 3. Get image dimensions (for denormalization and RLE)
        img_h, img_w = get_image_size(image_id)

        # 4. Determine the boxes
        if json_boxes is not None and len(json_boxes) > 0:
            boxes = json_boxes
        else:
            boxes = extract_boxes_from_text(explanation)
        # 4. [Modification] Extract boxes, denormalize, and update the explanation text
        # Old logic: raw_boxes = extract_boxes_from_text(explanation) -> denormalize -> (text not updated)
        # New logic: extract, denormalize, and update text in one step
        updated_explanation, final_boxes_exp = update_explanation_and_get_boxes(
            explanation, img_h, img_w
        )

        # 5. Denormalize coordinates
        final_boxes = denormalize_boxes(boxes, img_h, img_w)
        # if len(final_boxes) != len(final_boxes_exp):
        #     print(f"{image_name}, generated boxes do not match the boxes in exp")
        # 6. Generate RLE location
        # If label is 0 (real), theoretically no mask is needed, but generate an empty mask for format uniformity
        # If label is 1 but no boxes are detected, also generate an empty mask
        location_json_str = boxes_to_rle(final_boxes_exp, img_h, img_w)

        # 7. Build the result row
        results.append(
            {
                "image_name": image_name,
                "label": label,
                "location": location_json_str,
                "explanation": updated_explanation,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Simulated input data (load from file in actual use)
    # For demonstration, assume you have saved the provided sample as predictions.json
    if not os.path.exists(INPUT_JSON_PATH):
        print(
            f"Warning: {INPUT_JSON_PATH} not found. Please save your model output to this file."
        )
        # Create an empty file to avoid errors; comment this line out in actual runs
        # with open(INPUT_JSON_PATH, 'w', encoding='utf-8') as f: json.dump([], f)
    else:
        data = load_data(INPUT_JSON_PATH)
        df = process_data(data)

        # Save the CSV
        # quoting=csv.QUOTE_ALL ensures all fields are wrapped in quotes, matching the example format
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"Successfully generated {OUTPUT_CSV_PATH} with {len(df)} rows.")
        print(f"Sample row:\n{df.iloc[0].to_dict()}")
