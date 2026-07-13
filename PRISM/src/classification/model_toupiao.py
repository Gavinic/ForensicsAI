from collections import Counter
from glob import glob

import pandas as pd

# Get all CSV files
csv_files = [
    "./infer_result_testB_base.csv",
    "./infer_result_testB_large.csv",
    "./infer_result_testB_xlarge.csv",
]

# Store all prediction results
all_predictions = {}

# Read all files
for file in csv_files:
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        path = row.iloc[0]  # First column is the path
        label = row.iloc[1]  # Second column is the label

        if path not in all_predictions:
            all_predictions[path] = []
        all_predictions[path].append(label)

# Vote and generate results
results = []
for path, labels in all_predictions.items():
    final_label = Counter(labels).most_common(1)[0][0]
    results.append([path, final_label])

# Save results (keep two columns)
result_df = pd.DataFrame(results, columns=["Path", "Label"])
result_df.to_csv("voting_result.csv", index=False)

print(f"Voting complete! Processed {len(results)} samples")
print(f"Results saved to voting_result.csv")
