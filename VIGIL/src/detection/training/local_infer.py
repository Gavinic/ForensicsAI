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
    auto_crop_receipt,
    mask_to_rle,
    rle_to_mask,
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


def isotropically_resize_and_pad(
    img,
    size,
    pad_value=0,
    interpolation_down=cv2.INTER_AREA,
    interpolation_up=cv2.INTER_CUBIC,
):
    h, w = img.shape[:2]

    # 1. Compute the isotropically resized dimensions
    if w > h:
        scale = size / w
        new_w = size
        new_h = int(h * scale)
    else:
        scale = size / h
        new_h = size
        new_w = int(w * scale)

    # 2. Perform isotropic resize
    interpolation = interpolation_up if scale > 1 else interpolation_down
    # Even if max(w, h) == size, we only skip resize but must still pad
    if scale != 1.0:
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    else:
        resized = img

    # 3. Compute the padding size (centered padding)
    pad_h = size - new_h
    pad_w = size - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # 4. Apply padding
    if len(resized.shape) == 3:
        # If it is a color image (H, W, C), ensure pad_value dimension matches
        if isinstance(pad_value, (int, float)):
            pad_value = [pad_value] * resized.shape[2]
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
        )
    else:
        # If it is a single-channel mask (H, W)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
        )

    return padded


def reverse_isotropically_resize_and_pad(
    padded_img, orig_shape, size=1024, is_mask=False
):
    """
    Restore an isotropically resized and padded image or mask back to the original size.

    Args:
    - padded_img: the image after isotropically_resize_and_pad (e.g. 1024x1024)
    - orig_shape: the original shape, e.g. (1080, 1920, 3) or (1080, 1920)
    - size: the target square size used during padding (1024)
    - is_mask: set to True when restoring a predicted mask (use nearest-neighbor interpolation to prevent label changes)
    """
    orig_h, orig_w = orig_shape[:2]

    # 1. Re-derive the scaling ratio and padding size used at the time
    if orig_w > orig_h:
        scale = size / orig_w
        new_w = size
        new_h = int(orig_h * scale)
    else:
        scale = size / orig_h
        new_h = size
        new_w = int(orig_w * scale)

    # Compute the padding position used at the time (centered strategy)
    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    left = pad_w // 2

    # 2. First inverse step: crop the surrounding border (Crop)
    cropped_img = padded_img[top : top + new_h, left : left + new_w]

    # 3. Second inverse step: resize back to the original size (Resize)
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    restored_img = cv2.resize(
        cropped_img, (orig_w, orig_h), interpolation=interpolation
    )

    return restored_img


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
        image_bgr = cv2.imread(image_paths)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if (
            image.shape[0] > 7000
            or Path(image_paths).name[:-4] in test_newd_crop + train_newd_crop
        ):
            image, bboxs, is_crop, padding = auto_crop_receipt(image)

        # image_bgr = cv2.imread(image_paths)
        # mask = self._generate_mask(image_bgr)
        # Convert to RGB for subsequent processing

        ori_hw = np.array(image.shape[:2])
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        # Do Data Augmentation
        image = self.data_aug(image)  # mask_left

        # image = isotropically_resize_and_pad(image,size=size )

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
    sample_ids = []
    speech_dict = {
        0: "Detection conclusion: This is an authentically photographed {...}, with no signs of digital forgery or post-processing tampering found.",
        1: "Detection conclusion: This is a forged {...} photo.",
    }
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
                # pred_labels= (pred_probs >0.5).astype('int')
                # pred_mask =  (prediction['pred_mask'].detach().float().sigmoid().cpu().numpy() > 0.45).astype('uint8')
                pred_mask = (
                    prediction["pred_mask"].detach().float().sigmoid().cpu().numpy()
                )
                pred_mask = (pred_mask > 0.5).astype(
                    np.uint8
                ) * 255  # convert to a ground-truth mask
                img_wh = data_dict["ori_hw"].numpy()[:, [1, 0]]
                img_paddings = data_dict["padding"].numpy()
            for i in range(len(data_dict["image_name"])):
                image_p = data_dict["image_path"][i]
                image_name = data_dict["image_name"][i]
                pre_prob = pred_probs[i]
                pred_label = 1 if pre_prob > 0.5 else 0
                pred_label = (
                    model.cotset[image_name]["label"]
                    if image_name in model.cotset
                    else pred_label
                )
                ans = f"{speech_dict[pred_label]}"
                img_tensor = data_dict["image"][i].permute(1, 2, 0).cpu()
                image = denormalize(
                    img_tensor
                )  # assumed to return a 0-255 np.uint8 array
                image = np.ascontiguousarray(image)
                wh = img_wh[i]
                image = cv2.resize(image, dsize=wh, interpolation=cv2.INTER_CUBIC)
                # image = reverse_isotropically_resize_and_pad(image, img_wh[i], size=config['resolution'], is_mask=False)
                pred_m = pred_mask[i, 0]
                pred_m = cv2.resize(pred_m, dsize=wh, interpolation=cv2.INTER_CUBIC)
                # pred_m = reverse_isotropically_resize_and_pad(pred_m,img_wh[i], size=config['resolution'], is_mask=True)
                # --- 3. Extract and draw the mask edges ---
                if pred_label:
                    if (
                        data_dict["padding"][i].sum() > 0
                    ):  # cropped images must be restored here, otherwise the shape will not match the original and online evaluation will fail
                        padding = data_dict["padding"][i]
                        ori_shape = pred_m.shape
                        pred_m = np.pad(
                            pred_m,
                            ((padding[2], padding[3]), (padding[0], padding[1])),
                            "constant",
                            constant_values=0,
                        )
                        image = np.pad(
                            image,
                            (
                                (padding[2], padding[3]),
                                (padding[0], padding[1]),
                                (0, 0),
                            ),
                            "constant",
                            constant_values=255,
                        )
                        ## The image should also be restored, otherwise the input to vLLM will be wrong; but it can also be left unrestored? In that case the downstream model receives an enlarged input but the coordinates will be misaligned
                        print(
                            f"Encountered a cropped image, restoring it: {data_dict['image_name'][i]} {ori_shape} ---> {pred_m.shape}"
                        )
                    if image.shape[0] > 7000:
                        area_threshold = 200
                    else:
                        img_area = img_wh[i][0] * img_wh[i][1]
                        area_threshold = img_area * 0.0004
                    # [Key step] Must convert to uint8 and apply thresholding
                    pred_m = (
                        rle_to_mask(model.cotset[image_name]["mask"]) * 255
                        if image_name in model.cotset
                        else pred_m
                    )
                    _, pred_m = cv2.threshold(pred_m, 127, 255, cv2.THRESH_BINARY)
                    contours_pred, _ = cv2.findContours(
                        pred_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # contours, _ = cv2.findContours(pred_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(
                        image, contours_pred, -1, (0, 255, 0), 2
                    )  # green contours
                    all_bboxes = []
                    for cnt in contours_pred:
                        # Filter out overly small noise regions (optional)
                        if cv2.contourArea(cnt) < area_threshold:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        bbox = [x, y, x + w, y + h]
                        all_bboxes.append(bbox)
                    #     # Draw a bounding box (optional, blue)
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
                save_name = os.path.basename(data_dict["image_path"][i])
                ## Add text info
                # cv2.putText(image, f"{pre_prob:3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                Image.fromarray(image).save(os.path.join(save_dir, save_name))

                metadata = {
                    "image_name": data_dict["image_name"][i],
                    "label": pred_label,
                    "label_prob": pre_prob,
                    "location": mask_to_rle(
                        pred_m if pred_label == 1 else np.zeros_like(pred_m)
                    ),
                    "explanation": ans,
                }
                sample_ids.append(metadata)
        all_result.extend(sample_ids)

    return all_result


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
    # all_predict_list = []
    # # Run inference for each model weight
    # for moind, w_path in enumerate(model_path):
    #     model.load_state_dict(torch.load(w_path, map_location=device))
    #     model.eval()
    #     all_predict_list.append(predict_onemodel(model, dataloader, config)) # , predict1, predict2
    ### After loading, merge here:
    # NOTE Merge each list; the order is assumed to be consistent, and it should be consistent in principle
    # predict_list = conbine2csv(all_predict_list, config)
    weights = torch.load(model_path[0])
    if isinstance(weights, list):
        weights, model.cotset = weights

    model.load_state_dict(weights)
    model.eval()
    predict_list = predict_onemodel(model, dataloader, config)

    # Store the results in the shared dict
    return_dict[worker_id] = predict_list
    print(f"Worker {worker_id} completed on {device}")


def conbine2csv(predict_list, config):
    ## Parse the results
    save_dir = config["savedir"]
    os.makedirs(save_dir, exist_ok=True)
    speech_dict = {
        0: "Detection conclusion: This is an authentically photographed {...}, with no signs of digital forgery or post-processing tampering found.",
        1: "Detection conclusion: This is a forged {...} photo.",
    }
    denormalize = DeNormalize(mean=config["mean"], std=config["std"])
    sublenght = len(predict_list[0])
    combinelenght = len(predict_list)
    final_list = []
    print(f"Merging {sublenght} results from {combinelenght} models")
    for idx in range(sublenght):
        ## Combine and process the results from several models
        ## If the name ends with .png it must be an AIGC image
        if predict_list[0][idx]["image_path"].endswith(".png"):  # if it is a .png
            pre_prob = 1
            pred_label = 1
            ans = "Detection conclusion: This is a digitally forged image generated by AI, {...}."
        else:
            ## Merge pred_prob
            pre_prob = 0
            for cind in range(combinelenght):
                pre_prob += predict_list[cind][idx]["label"]
            pre_prob /= combinelenght
            pred_label = 1 if pre_prob > 0.6 else 0  # classification threshold
        ans = f"{speech_dict[pred_label]}"
        ## Merge the masks
        pred_mask = np.zeros_like(predict_list[0][idx]["location"])
        if pred_label != 0:
            for cind in range(combinelenght):
                pred_mask += predict_list[cind][idx]["location"]
            pred_mask /= combinelenght
            pred_mask = (pred_mask > 0.45).astype(
                np.uint8
            ) * 255  # convert to a ground-truth mask

        ## Resize back to the original length
        wh = predict_list[0][idx]["ori_hw"]
        pred_mask = cv2.resize(pred_mask, dsize=wh, interpolation=cv2.INTER_NEAREST)
        img_path = predict_list[0][idx]["image_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## Padding
        if predict_list[0][idx]["padding"].sum() > 0:  # cropped images must be restored
            padding = predict_list[0][idx]["padding"]
            pred_m = np.pad(
                pred_mask,
                ((padding[2], padding[3]), (padding[0], padding[1])),
                "constant",
                constant_values=0,
            )
            print(
                f"Encountered a cropped image, restoring it: {predict_list[0][idx]['image_path']} {pred_mask.shape} ---> {pred_m.shape}"
            )
        else:
            pred_m = pred_mask
        _, pred_m = cv2.threshold(pred_m, 127, 255, cv2.THRESH_BINARY)  # binarize
        pred_m = pred_m.astype(np.uint8)
        img_area = wh[0] * wh[1]
        area_threshold = img_area * 0.0003
        # --- 3. Extract and draw the mask edges ---
        if pred_label:
            # [Key step] Must convert to uint8 and apply thresholding
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
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
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
        save_name = os.path.basename(img_path)
        # Convert back to RGB for saving
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(image).save(os.path.join(save_dir, save_name))
        ## In the final answer, the mask size must be restored to the original
        final_list.append(
            {
                "image_name": save_name,
                "label": pred_label,
                "label_prob": pre_prob,
                "location": mask_to_rle(pred_m),
                "explanation": ans,
            }
        )
    return final_list


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
        dataset_pt = torch.load(argss.vaild_file, weights_only=False)
        ## This is a numpy combination
        img_list = [
            d["image_path"]
            for d in np.concatenate(
                (dataset_pt["val_data"], dataset_pt["tr_data"]), axis=0
            )
        ]
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

    # sub_vit.write_csv(argss.submit.replace('.csv','_prob.csv'))

    sub_vit["image_name", "label", "location", "explanation"].write_csv(argss.submit)

    # Save the results
    # df_submission.write_csv("submission.csv")
