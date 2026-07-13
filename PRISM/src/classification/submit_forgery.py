import os

import pandas as pd

result_file = r"./voting_result.csv"

data = {"image_name": [], "label": [], "location": [], "explanation": []}
df = pd.read_csv(result_file)

for i, row in df.iterrows():
    if pd.isna(row["Label"]):
        continue
    image_name = os.path.basename(row["Path"])
    data["image_name"].append(image_name)
    data["label"].append(row["Label"])
    data["location"].append("")
    data["explanation"].append("")

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("./submit_csv_cls.csv", index=False, encoding="utf-8-sig")
