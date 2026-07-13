import argparse
import os
import sys

# Create the argument parser
parser = argparse.ArgumentParser(
    description="List files in directory and save to test2.txt"
)
parser.add_argument("--input_path", "-i", required=True, help="Input directory path")
parser.add_argument(
    "--output_file",
    "-o",
    default="test2.txt",
    help="Output file path (default: test2.txt)",
)

args = parser.parse_args()

# Get arguments
path = args.input_path
save_file = args.output_file

# Check whether the input path exists
if not os.path.exists(path):
    print(f"Error: Input path does not exist: {path}")
    sys.exit(1)

# Write to file
with open(save_file, "w") as fw:
    for f in os.listdir(os.path.join(path, "ForgeryAnalysis_Stage_2_Test/Image")):
        fw.write(os.path.join(path, "ForgeryAnalysis_Stage_2_Test/Image", f) + "\n")

print(f"Successfully wrote file list to {save_file}")
print(f"Total files: {len(os.listdir(path))}")
