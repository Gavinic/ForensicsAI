import os
import random

fake_data_path = r"data/ForgeryAnalysis_Stage_1_Train/Black/Image"
right_data_path = r"data/ForgeryAnalysis_Stage_1_Train/White/Image"

files = []
files.extend(
    [os.path.join(fake_data_path, f) + ",1" for f in os.listdir(fake_data_path)]
)
files.extend(
    [os.path.join(right_data_path, f) + ",0" for f in os.listdir(right_data_path)]
)

print(len(files))
print(files[:5])
random.shuffle(files)
print(files[:5])

train_files = files[:]
test_files = files[950:]

random.shuffle(train_files)

with open("train.txt", "w") as fw:
    for ft in train_files:
        fw.write(ft + "\n")

with open("val.txt", "w") as fw:
    for ft in test_files:
        fw.write(ft + "\n")
