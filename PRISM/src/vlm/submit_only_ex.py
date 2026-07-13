import json
import os

import numpy as np
import pandas as pd


def load_mapping(test_json_path):
    """
    Load the image_id to image_name mapping from testB.json.
    """
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    # Build the image_id to image_name mapping dictionary
    id_to_name = {}
    for img_info in test_data["images"]:
        image_id = img_info["id"]
        # Get the file name (without the path)
        if "file_name" in img_info:
            image_name = os.path.basename(img_info["file_name"])
        elif "image_name" in img_info:
            image_name = os.path.basename(img_info["image_name"])
        else:
            image_name = f"{image_id}.jpg"
        id_to_name[image_id] = image_name

    return id_to_name


def load_response_mapping(response_files_dir):
    """
    Load all response files and build a mapping from image path to response content.
    """
    path_to_response = {}
    response_files = os.listdir(response_files_dir)

    print(f"Loading response files...")
    for file in response_files:
        file_path = os.path.join(response_files_dir, file)
        with open(file_path, "r") as fr:
            for line_idx, line in enumerate(fr.readlines()):
                json_data = json.loads(line.strip())

                # Extract the image path and response content
                if "images" in json_data and len(json_data["images"]) > 0:
                    image_path = json_data["images"][0].get("path", "")

                    # Get the response content
                    if "response" in json_data:
                        response_text = json_data["response"]
                    elif "messages" in json_data and len(json_data["messages"]) > 1:
                        response_text = json_data["messages"][1].get("content", "")
                    else:
                        response_text = ""

                    # Use the image path as the key to store the response
                    if image_path:
                        path_to_response[image_path] = response_text

    print(f"Loaded {len(path_to_response)} response entries")
    return path_to_response


def extract_explanation_from_response(response_text):
    """
    Extract the explanation field from the response text.
    The response text may be a JSON string containing label and explanation fields.
    """
    if not response_text:
        return ""

    try:
        # First remove any <think> tags that may be present
        response_text = response_text.replace("<think>\n\n</think>\n\n", "")

        # Escape double quotes so JSON can be parsed
        # Replace double quotes in the string with escaped double quotes
        escaped_text = response_text.replace('"', '\\"')

        # Be careful: if the string already contains escapes, avoid double-escaping
        # Simple approach here: escape all double quotes, then JSON parsing handles it correctly

        # Try to parse as JSON
        response_json = json.loads(escaped_text)

        # Extract the explanation field
        if "explanation" in response_json:
            return response_json["explanation"]
        else:
            # If there is no explanation field, return the original text
            return response_text

    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Text attempted to parse: {response_text[:200]}...")

        # If escaping all double quotes failed, try another approach
        try:
            # Method 2: only escape double quotes inside the string
            # Assume the JSON format is {"label": 1, "explanation": "text with \"quotes\""}
            # Smarter handling is needed here
            import re

            # Match the value of the explanation field and escape its inner double quotes
            pattern = r'"explanation":\s*"([^"]*(?:\\"[^"]*)*)"'
            match = re.search(pattern, response_text)
            if match:
                explanation_text = match.group(1)
                # Restore the escaped double quotes
                explanation_text = explanation_text.replace('\\"', '"')
                return explanation_text
        except:
            pass

        # If everything failed, return the original text
        return response_text


def build_image_id_to_path_mapping(test_json_path):
    """
    Build an image_id to image path mapping from testB.json.
    """
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    id_to_path = {}
    for img_info in test_data["images"]:
        image_id = img_info["id"]
        # Get the full image path (if available)
        if "file_name" in img_info:
            # The full path may need to be constructed based on the actual situation
            image_path = img_info["file_name"]
        else:
            image_path = f"unknown_path_{image_id}.jpg"
        id_to_path[image_id] = image_path

    return id_to_path


if __name__ == "__main__":
    # File path configuration
    response_files_dir = (
        r"../../models/Qwen3.5-9B-Turn/v1-20260321-220639/checkpoint-310/infer_result"
    )
    test_json_path = r"../../data/testB.json"
    save_path = r"./submit_csv_ex.csv"

    # Step 1: load testB.json and build image_id to image_name and path mappings
    print("=" * 50)
    print("Loading testB.json mappings...")
    id_to_name = load_mapping(test_json_path)
    id_to_path = build_image_id_to_path_mapping(test_json_path)
    all_image_ids = set(id_to_name.keys())
    print(f"testB.json contains {len(all_image_ids)} images in total")

    # Step 2: load all response files and build an image path to response mapping
    print("=" * 50)
    path_to_response = load_response_mapping(response_files_dir)

    # Step 3: create the final DataFrame
    print("=" * 50)
    print("Creating the CSV file...")

    data = {"image_name": [], "label": [], "location": [], "explanation": []}

    matched_count = 0
    unmatched_count = 0
    json_parse_error_count = 0

    # Iterate over all image IDs
    for image_id in all_image_ids:
        image_name = id_to_name.get(image_id, f"{image_id}.jpg")

        # Try to get the corresponding image path
        image_path = id_to_path.get(image_id, "")

        # Look up the corresponding response from path_to_response
        if image_path and image_path in path_to_response:
            response_text = path_to_response[image_path]
            matched_count += 1

            # Extract the explanation field
            explanation = extract_explanation_from_response(response_text)

            # Check whether the explanation was successfully parsed
            if explanation == response_text and response_text:
                # If parsing failed but the response is non-empty, it may be a parse error
                json_parse_error_count += 1
                print(
                    f"Warning: the response for image {image_name} is not standard JSON format, using the original text"
                )
        else:
            # If no matching response was found, try matching by file name
            found = False
            basename_image = os.path.basename(image_name)
            for path, response_text in path_to_response.items():
                if (
                    os.path.basename(path) == basename_image
                    or os.path.basename(path) == image_name
                ):
                    # Extract the explanation field
                    explanation = extract_explanation_from_response(response_text)
                    found = True
                    matched_count += 1

                    # Check whether the explanation was successfully parsed
                    if explanation == response_text and response_text:
                        json_parse_error_count += 1
                        print(
                            f"Warning: the response for image {image_name} is not standard JSON format, using the original text"
                        )

                    print(
                        f"Matched by file name: {basename_image} -> {os.path.basename(path)}"
                    )
                    break

            if not found:
                explanation = ""  # Empty string as a placeholder
                unmatched_count += 1
                print(
                    f"Warning: no response content found for image {image_name} (ID: {image_id})"
                )

        # Remove <think> tags (if still present)
        explanation = explanation.replace("<think>\n\n</think>\n\n", "")

        data["image_name"].append(image_name)
        data["label"].append(0)  # Set all labels to 0
        data["location"].append("")  # Set all locations to empty strings
        data["explanation"].append(explanation)

    # Step 4: create the DataFrame and save it
    print("=" * 50)
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\nProcessing complete!")
    print(f"Processed {len(data['image_name'])} images in total")
    print(f"Images with a matched response: {matched_count}")
    print(f"Images without a response: {unmatched_count}")
    print(
        f"Images with JSON parse errors or non-standard format: {json_parse_error_count}"
    )
    print(f"Result saved to: {save_path}")

    # Verify the total count
    if len(data["image_name"]) == len(all_image_ids):
        print(f"\nSuccess: processed all {len(all_image_ids)} images")
    else:
        print(
            f"\nError: only processed {len(data['image_name'])} images, but there should be {len(all_image_ids)}"
        )

    # Print statistics
    forgery_count = sum(1 for label in data["label"] if label == 1)
    non_forgery_count = len(data["label"]) - forgery_count

    print(f"\nStatistics:")
    print(f"Images with forgery: {forgery_count}")
    print(f"Images without forgery: {non_forgery_count}")

    # Show a sample of the first few rows
    print("\n" + "=" * 50)
    print("Data sample (first 5 rows):")
    print(df.head())

    # Show a sample of the first explanation (if any)
    if len(df) > 0 and df["explanation"].iloc[0]:
        print("\n" + "=" * 50)
        print("First explanation sample:")
        print(
            df["explanation"].iloc[0][:200] + "..."
            if len(df["explanation"].iloc[0]) > 200
            else df["explanation"].iloc[0]
        )
