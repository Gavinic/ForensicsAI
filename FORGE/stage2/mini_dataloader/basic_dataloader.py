import os

import numpy as np

np.random.seed(42)
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Lambda
from tqdm import tqdm

from .post_process import (
    gaussian_blur,
    gaussian_noise,
    horizontal_flip,
    jpeg_compression,
    resize,
    rotate,
    vertical_flip,
)


def read_shuffle(txt):
    with open(txt, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.read().splitlines() if line.strip()]
    return lines


def process(
    forgery, mask, augment_prob=0, enable_aug_types=None, intensity=None, mode="train"
):
    if intensity is None:
        intensity = {
            "rates": None,
            "qfs": None,
            "sds": None,
            "ksizes": None,
            "crop_size": None,
        }
    # _/_/_/ Data Augmentation _/_/_/
    if mode == "train":
        num_aug_types = np.random.choice(4, 1)
        aug_types = np.random.choice([0, 1, 2], num_aug_types, replace=False)
        for aug_type in aug_types:
            if aug_type == 0:
                forgery, mask = horizontal_flip(forgery, mask, p=augment_prob)
            elif aug_type == 1:
                forgery, mask = vertical_flip(forgery, mask, p=augment_prob)
            elif aug_type == 2:
                forgery, mask = rotate(forgery, mask, p=augment_prob)

        if enable_aug_types is not None and len(enable_aug_types) > 0:
            aug_types = np.random.choice(enable_aug_types, 1, replace=False)
            for aug_type in aug_types:
                if aug_type == 3:
                    forgery, mask = resize(
                        forgery, mask, rates=intensity["rates"], p=augment_prob
                    )
                if aug_type == 4:
                    forgery = jpeg_compression(
                        forgery, qfs=intensity["qfs"], p=augment_prob
                    )
                elif aug_type == 5:
                    forgery = gaussian_noise(
                        forgery, sds=intensity["sds"], p=augment_prob
                    )
                elif aug_type == 6:
                    forgery = gaussian_blur(
                        forgery, ksizes=intensity["ksizes"], p=augment_prob
                    )
    else:
        for aug_type in enable_aug_types or []:
            if aug_type == 3:
                forgery, mask = resize(
                    forgery, mask, rates=intensity["rates"], p=augment_prob
                )
            if aug_type == 4:
                forgery = jpeg_compression(
                    forgery, qfs=intensity["qfs"], p=augment_prob
                )
            elif aug_type == 5:
                forgery = gaussian_noise(forgery, sds=intensity["sds"], p=augment_prob)
            elif aug_type == 6:
                forgery = gaussian_blur(
                    forgery, ksizes=intensity["ksizes"], p=augment_prob
                )
    return forgery, mask


class BasicDataloader(Dataset):  # Read list, resize
    def __init__(
        self,
        dataset_list=None,
        sample_records=None,
        adv_list=None,
        input_size=1024,
        normalize_type=2,
        augment_prob=0,
        enable_aug_types=None,
        intensity=None,
        mode="train",
    ):
        if dataset_list is None:
            dataset_list = []
        self.adv_list = adv_list
        self.sam_input_size = input_size
        self.normalize_type = normalize_type
        self.augment_prob = augment_prob
        self.enable_aug_types = enable_aug_types
        self.intensity = (
            intensity
            if intensity is not None
            else {
                "rates": None,
                "qfs": None,
                "sds": None,
                "ksizes": None,
                "crop_size": None,
            }
        )
        self.mode = mode

        # Initialize the storage dict used to hold all merged data keyed by field
        self.data = {
            "forgery_path": [],
            "gt_mask_path": [],
            "forged_label": [],
            "adv_label": [],
        }

        if sample_records is not None:
            for record in sample_records:
                self.data["forgery_path"].append(record["forgery_path"])
                self.data["gt_mask_path"].append(record.get("gt_mask_path", ""))
                self.data["forged_label"].append(int(record.get("forged_label", 0)))
                self.data["adv_label"].append(int(record.get("adv_label", 0)))
        else:
            for path, name, ratio, load_times, forged_label, adv_label in dataset_list:
                lines = read_shuffle(txt=os.path.join(path, name))
                for idx in range(load_times):
                    length = int(ratio * len(lines))
                    t = tqdm(lines[:length])
                    for line in t:
                        if "," in line:
                            forgery_path, gt_mask_path = line.split(",")[-2:]
                            forgery_path = os.path.join(path, forgery_path)
                            gt_mask_path = os.path.join(path, gt_mask_path)
                        else:
                            forgery_path, gt_mask_path = line, ""
                            forgery_path = os.path.join(path, forgery_path)
                        self.data["forgery_path"].append(forgery_path)
                        self.data["gt_mask_path"].append(gt_mask_path)
                        self.data["forged_label"].append(forged_label)
                        self.data["adv_label"].append(adv_label)
                        t.set_description(f"Load {idx+1}/{load_times} {path}/{name}")
        print(f"Load data.-> {len(self.data['forgery_path'])}")

        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        # transform
        def identity(x):
            return x

        if self.normalize_type == 0:
            normalization = Lambda(identity)
        elif self.normalize_type == 1:
            normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.normalize_type == 2:
            normalization = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        self.image_transform = transforms.Compose(
            [np.float32, transforms.ToTensor(), normalization]
        )
        self.mask_transform = transforms.Compose(
            [
                np.float32,
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        forgery = cv2.imread(self.data["forgery_path"][idx])  # (1536, 2048, 3)
        if forgery is None:
            raise FileNotFoundError(
                f"Image not found or unreadable: {self.data['forgery_path'][idx]}"
            )
        if self.data["gt_mask_path"][idx] == "":
            h, w = forgery.shape[:2]
            gt_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            gt_mask = cv2.imread(self.data["gt_mask_path"][idx], 0)  # (1536, 2048)
            if gt_mask is None:
                raise FileNotFoundError(
                    f"Mask not found or unreadable: {self.data['gt_mask_path'][idx]}"
                )

        forgery, gt_mask = process(
            forgery,
            gt_mask,
            augment_prob=self.augment_prob,
            enable_aug_types=self.enable_aug_types,
            intensity=self.intensity,
            mode=self.mode,
        )

        if self.sam_input_size is not None:
            forgery = cv2.resize(
                forgery, (self.sam_input_size, self.sam_input_size), cv2.INTER_AREA
            )

        forgery = forgery[:, :, ::-1].astype(np.float32) / 255.0

        if self.sam_input_size is not None:
            gt_mask = cv2.resize(
                gt_mask,
                (self.sam_input_size // 4, self.sam_input_size // 4),
                cv2.INTER_NEAREST,
            )
        gt_mask = np.where(gt_mask > 127, 255, 0).astype(np.float32) / 255.0

        forged_label = torch.as_tensor(self.data["forged_label"][idx])
        adv_label = torch.as_tensor(self.data["adv_label"][idx])
        return (
            self.image_transform(forgery),
            self.mask_transform(gt_mask),
            forged_label,
            adv_label,
        )

    def __len__(self):
        return len(self.data["forgery_path"])

    def get_name(self, idx):
        return self.data["forgery_path"][idx]

    def get_image_gt(self, idx):
        forgery = cv2.imread(self.data["forgery_path"][idx])  # (1536, 2048, 3)
        if forgery is None:
            raise FileNotFoundError(
                f"Image not found or unreadable: {self.data['forgery_path'][idx]}"
            )
        if self.data["gt_mask_path"][idx] == "":
            h, w = forgery.shape[:2]
            gt_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            gt_mask = cv2.imread(self.data["gt_mask_path"][idx], 0)  # (1536, 2048)
            if gt_mask is None:
                raise FileNotFoundError(
                    f"Mask not found or unreadable: {self.data['gt_mask_path'][idx]}"
                )
        forgery = forgery[:, :, ::-1].astype(np.float32) / 255.0
        gt_mask = gt_mask.astype(np.float32) / 255.0
        return forgery, gt_mask
