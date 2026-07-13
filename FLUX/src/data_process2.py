# This script merges the jsonl files generated from the provided training set and the additional training set

#!/usr/bin/env python3
"""
Merge two JSONL files.
Each line of a JSONL file is a JSON object (dict).
"""

import json
import os
import sys


def merge_jsonl_files(file1_path, file2_path, output_path):
    """
    Merge two JSONL files.

    Args:
        file1_path: first JSONL file path
        file2_path: second JSONL file path
        output_path: output file path
    """
    total_count = 0

    try:
        # Check whether input files exist
        if not os.path.exists(file1_path):
            print(f"Error: file {file1_path} does not exist")
            return False

        if not os.path.exists(file2_path):
            print(f"Error: file {file2_path} does not exist")
            return False

        # Open the output file
        with open(output_path, "w", encoding="utf-8") as outfile:
            # Process the first file
            print(f"Reading the first file: {file1_path}")
            with open(file1_path, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            # Validate JSON format (optional)
                            json.loads(line)
                            outfile.write(line + "\n")
                            total_count += 1
                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: line {line_num} of {file1_path} is not valid JSON, skipped: {e}"
                            )

            # Process the second file
            print(f"Reading the second file: {file2_path}")
            with open(file2_path, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            # Validate JSON format (optional)
                            json.loads(line)
                            outfile.write(line + "\n")
                            total_count += 1
                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: line {line_num} of {file2_path} is not valid JSON, skipped: {e}"
                            )

        print(f"Success! Merged {total_count} records into {output_path}")
        return True

    except Exception as e:
        print(f"Error: exception occurred during merge: {e}")
        return False


def main():
    # Option 1: specify file paths directly
    file1 = "./data/data_json.jsonl"
    file2 = "./data/extra_grounding_data.jsonl"
    output = "./data/data.jsonl"

    print(f"Merging {file1} and {file2} into {output}")
    merge_jsonl_files(file1, file2, output)


if __name__ == "__main__":
    main()
