import argparse
import gc
import glob
import logging
import math
import os

# torch.backends.cudnn.enabled = False
import pickle
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from types import MethodType

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import polars as pl
import timm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from dataset.albu import DeNormalize
from dataset.idbm_dataset import (
    IdbmDataset,
    auto_crop_receipt,
    mask_to_rle,
    test_newd_crop,
    train_newd_crop,
)
from detectors import DETECTOR
from PIL import Image
from timm.layers import Mlp
from timm.models.eva import EvaAttention
from torch.amp import GradScaler, autocast
from torch.utils import data
from torch.utils.checkpoint import checkpoint
from torchvision import transforms as T
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from segment_proc import (
#     load_vegannmodel,
#     prediction_XG_SVM,
#     get_preprocessing_fn
# )


class IdbmDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, data_list, transform, config=None, mode="train"):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.data_list = data_list
        self.config = config
        self.mode = mode
        self.lenght = len(self.data_list)

        self.transform = transform

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Image.fromarray(np.array(img, dtype=np.uint8))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config["mean"]
        std = self.config["std"]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, mask=None, augmentation_seed=None):
        """Apply data augmentation"""

        kwargs = {"image": img}  #'mask': mask
        transformed = self.transform(**kwargs)
        augmented_img = transformed["image"]

        return augmented_img  # , augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_list[index]
        # Load the image
        is_crop = False
        padding = [0, 0, 0, 0]
        if Path(image_paths).name[:-4] in test_newd_crop + train_newd_crop:
            image, bboxs, is_crop, padding = auto_crop_receipt(image_paths)
        else:
            image_bgr = cv2.imread(image_paths)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_bgr = cv2.imread(image_paths)
        # mask = self._generate_mask(image_bgr)
        # Convert to RGB for subsequent processing

        ori_hw = np.array(image.shape[:2])
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        # Do Data Augmentation
        image = self.data_aug(image)  # mask_left
        image_trans = self.normalize(self.to_tensor(image))

        # mask_left = torch.tensor(mask_left, dtype=torch.float32) # H,W,2
        # mask_right = torch.tensor(mask_right, dtype=torch.float32)
        # label =  np.log1p(label)
        data_dict = {
            "image_path": str(image_paths),
            "image_name": Path(image_paths).name,
            "image": image_trans,
            "ori_hw": ori_hw,
            "padding": np.array(padding),
        }
        return data_dict

    def __len__(self):

        return self.lenght


def predict_onemodel(model, dataloader, config):
    model.eval()
    all_result = []
    # all_result1 = []
    # all_result2 = []
    speech_dict = {
        0: "Detection conclusion: This is an authentically photographed {...}, with no signs of digital forgery or post-processing tampering found.",
        1: "Detection conclusion: This is a forged {...} photo.",
    }
    sample_ids = []
    denormalize = DeNormalize(mean=config["mean"], std=config["std"])

    save_dir = config["savedir"]
    os.makedirs(save_dir, exist_ok=True)
    for data_dict in tqdm(dataloader):
        sample_ids = []
        with autocast("cuda", dtype=torch.bfloat16):
            for key in data_dict.keys():
                if data_dict[key] != None and key not in [
                    "image_path",
                    "ori_hw",
                    "image_name",
                    "padding",
                ]:
                    data_dict[key] = data_dict[key].to(config["device"])
            with torch.no_grad():
                prediction = model(data_dict)
                pred_probs = (
                    prediction["pred_label"].detach().float().sigmoid().cpu().numpy()
                )
                pred_labels = (pred_probs > 0.5).astype("int")
                pred_mask = (
                    prediction["pred_mask"].detach().float().sigmoid().cpu().numpy()
                    > 0.45
                ).astype("uint8")
                img_wh = data_dict["ori_hw"].numpy()[:, [1, 0]]
                img_paddings = data_dict["padding"].numpy()
        ## Parse the results
        # print(data_dict['ori_hw'])
        for i in range(len(data_dict["image_name"])):
            ## If the name ends with .png it must be an AIGC image
            if data_dict["image_name"][i].endswith(".png"):
                pred_labels[i] = 1
                ans = "Detection conclusion: This is a digitally forged image generated by AI."
            else:
                ans = f"{speech_dict[pred_labels[i]]}"
            img_tensor = data_dict["image"][i].permute(1, 2, 0).cpu()
            image = denormalize(img_tensor)  # assumed to return a 0-255 np.uint8 array
            image = np.ascontiguousarray(image)
            ## Resize back to the original length
            wh = img_wh[i]
            image = cv2.resize(image, dsize=wh, interpolation=cv2.INTER_CUBIC)
            pred_m = pred_mask[i, 0]
            pred_m = cv2.resize(pred_m, dsize=wh, interpolation=cv2.INTER_CUBIC)
            img_area = wh[0] * wh[1]
            area_threshold = img_area * 0.0004
            # --- 3. Extract and draw the mask edges ---
            if pred_labels[i]:
                # [Key step] Must convert to uint8 and apply thresholding
                _, pred_m = cv2.threshold(
                    (pred_m * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY
                )
                contours_pred, _ = cv2.findContours(
                    pred_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                all_bboxes = []
                for cnt in contours_pred:
                    # Filter out overly small noise regions (optional)
                    if cv2.contourArea(cnt) < area_threshold:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = [x, y, x + w, y + h]
                    all_bboxes.append(bbox)

                    # Draw red edges
                    # cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
                    # Draw a bounding box (optional, blue)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # cv2.drawContours(image, contours_pred, -1, (0, 0, 255), 2) # red edges
                if len(all_bboxes[:6]) == 1:
                    ans = (
                        ans
                        + f"There is one key tampered region in the image, located at coordinates: {all_bboxes[0]}, content is.."
                    )
                elif len(all_bboxes[:6]) < 5:
                    ans = (
                        ans
                        + f"There are {len(all_bboxes[:6])} forged regions; key tampered regions include:{','.join([str(bbox) for bbox in all_bboxes[:6]])}"
                    )
                else:
                    ans = (
                        ans
                        + f"There are multiple forged regions, with coordinates:{','.join([str(bbox) for bbox in all_bboxes[:6]])}"
                    )

            # --- 4. Draw text info ---
            # prob_val = pred_probs[i]
            # text_pred = f"Pred Prob: {prob_val:.3f}"
            # cv2.putText(image, text_pred, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- 5. Save ---
            save_name = os.path.basename(data_dict["image_path"][i])
            # Convert back to RGB for saving
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(image).save(os.path.join(save_dir, save_name))
            ## In the final answer, the mask size must be restored to the original
            if img_paddings[i].sum() > 0:
                padding = img_paddings[i]
                pred_m = np.pad(
                    pred_m,
                    ((padding[2], padding[3]), (padding[0], padding[1])),
                    "constant",
                    constant_values=0,
                )
                print(
                    f"Encountered a cropped image, restoring it: {data_dict['image_name'][i]} {image.shape} ---> {pred_m.shape}"
                )

            sample_ids.append(
                {
                    "image_name": data_dict["image_name"][i],
                    "label": pred_labels[i],
                    "location": mask_to_rle(
                        pred_m if pred_labels[i] == 1 else np.zeros_like(pred_m)
                    ),
                    "explanation": ans,
                }
            )
        all_result.extend(sample_ids)

    return all_result


# , \
# pl.from_dict({"sample_id":sample_ids,"target":all_result1}), \
# pl.from_dict({"sample_id":sample_ids,"target":all_result2})


def inference_worker(
    gpu_id, img_list_subset, model_path, config, return_dict, worker_id
):
    """
    Inference worker on a single GPU

    Args:
        gpu_id: GPU device ID (0 or 1)
        img_list_subset: list of images assigned to this GPU
        model_path: list of model weight paths
        config: config dict
        return_dict: multiprocess shared dict used to return results
        worker_id: worker identifier
    """
    print("step-----")
    # Set the GPU used by the current process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    config["device"] = device
    if len(img_list_subset) < 1:
        print("no image in this gpu, return None")
        return_dict[worker_id] = None
        return

    # ===== Performance optimization settings =====
    # torch.backends.cudnn.benchmark = False #True  # enable CuDNN auto-tuning
    # torch.backends.cudnn.deterministic = False  # allow non-deterministic algorithms (faster)

    print(f"Worker {worker_id} starting on {device} with {len(img_list_subset)} images")
    resize = config["resolution"]
    trans = A.Compose(
        [
            # A.HorizontalFlip(p=1),
            A.Resize(height=resize, width=resize, p=1)
        ],
    )
    # Create the dataset and data loader
    dataset = IdbmDataset(img_list_subset, transform=trans, config=config)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=2,  # reduce the number of workers because there are two processes
        shuffle=False,
        pin_memory=True,
    )

    # Load the model
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)
    model = model.to(device)
    model.eval()
    # Run inference for each model weight
    for moind, w_path in enumerate(model_path):
        model.load_state_dict(torch.load(w_path, map_location=device))
        model.eval()
        predict_list = predict_onemodel(
            model, dataloader, config
        )  # , predict1, predict2

    # Store the results in the shared dict
    return_dict[worker_id] = predict_list
    print(f"Worker {worker_id} completed on {device}")


def parallel_inference(img_list, model_path, config, num_gpus=2):
    """
    Run parallel inference across multiple GPUs

    Args:
        img_list: the full image list
        model_path: list of model weight paths
        config: config dict
        num_gpus: number of GPUs to use

    Returns:
        The merged polars DataFrame
    """

    # Set the multiprocess start method
    mp.set_start_method("spawn", force=True)

    # Split the image list into num_gpus chunks
    chunk_size = len(img_list) // num_gpus
    img_list_chunks = [
        img_list[
            i * chunk_size : (i + 1) * chunk_size if i < num_gpus - 1 else len(img_list)
        ]
        for i in range(num_gpus)
    ]

    print(f"Split {len(img_list)} images into {num_gpus} chunks:")
    for i, chunk in enumerate(img_list_chunks):
        print(f"  GPU {i}: {len(chunk)} images")

    # Create a multiprocess manager and shared dict
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create the processes
    processes = []
    for gpu_id in range(num_gpus):
        print("****", gpu_id)
        p = mp.Process(
            target=inference_worker,
            args=(
                gpu_id,
                img_list_chunks[gpu_id],
                model_path,
                config,
                return_dict,
                gpu_id,
            ),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Merge the results
    print("Merging results from all GPUs...")
    df_results = []
    # print(return_dict)
    for i in range(num_gpus):
        tmp = return_dict[i]
        if tmp is not None:
            df_results.extend(tmp)

    print(f"Final result shape: {len(df_results)}")
    return df_results


def load_csv2list(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: Detected Long Format. Pivoting...")
    # For submission file, we just need unique image IDs
    unique_ids = df["image_path"].unique()
    print(f"nums: {len(unique_ids)}")
    return list(unique_ids)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_path", type=str, default="*.pth")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--savedir", type=str, default="./vis_test")
    parser.add_argument("--vaild_file", type=str, default="")
    parser.add_argument("--submit", type=str, default="vit_dinov3_LB_0598.csv")
    args = parser.parse_args()
    return args


# # # Usage example
if __name__ == "__main__":
    argss = args()
    # Ensure two GPUs are available
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if argss.vaild_file:
        ## If this is the validation-mode set
        vaild_data = torch.load(argss.vaild_file, weights_only=False)[
            "val_data"
        ]  # this is a .pt file
        ## This is a numpy combination
        img_list = [d["image_path"] for d in vaild_data]
    else:
        img_list = glob.glob(argss.test_path)

    with open(argss.config, "r") as f:
        config = yaml.safe_load(f)
    config["savedir"] = argss.savedir
    model_path = glob.glob(argss.model_path)
    print(model_path)
    # Run parallel inference
    df_submission_vit = parallel_inference(img_list, model_path, config, num_gpus=1)
    sub_vit = pl.from_dicts(df_submission_vit)
    sub_vit.write_csv(argss.submit)

    # Save the results
    # df_submission.write_csv("submission.csv")
