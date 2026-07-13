import glob
import json
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import tile
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from utils.compare import compare, count
from utils.create_dir import create_dir
from utils.dataset_n import Garbage_DatasetInfer
from utils.log import get_logger
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.cuda.empty_cache()
model_state = torch.load(
    "../../models/classification/max_vit_base_forgery_fake_best.pth", map_location="cpu"
)
model = timm.create_model("maxvit_base_tf_512.in21k_ft_in1k", pretrained=False)
num_ftrs = model.head.fc.in_features
model.head.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(model_state)
model = nn.DataParallel(model).cuda()

test_dir = "./test2.txt"
image_datasets = Garbage_DatasetInfer(test_dir)
dataset_loaders = torch.utils.data.DataLoader(
    image_datasets, batch_size=8, shuffle=False, num_workers=4
)
model.eval()
idx = 1
result = {"Path": [], "Label": []}
softmax = nn.Softmax(dim=1)
for data in dataset_loaders:
    print(idx)
    inputs, paths = data
    inputs = inputs.cuda()
    outputs = model(inputs)
    scores = softmax(outputs)
    predicted_class = torch.argmax(scores, axis=1)
    # print(scores, predicted_class)
    idx += 1
    for p, label in zip(paths, predicted_class):
        result["Path"].append(p)
        result["Label"].append(label.item())

df = pd.DataFrame(result)
df.to_csv("infer_result_testB_base.csv", index=False)
