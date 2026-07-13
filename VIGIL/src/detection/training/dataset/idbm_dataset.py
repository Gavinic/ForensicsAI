import sys

sys.path.append(".")

import ast
import glob
import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import polars as pl
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils import data
from torchvision import transforms as T

from .albu import (
    IsotropicResize,
    isotropically_resize_and_pad,
    isotropically_resize_image,
)


# Convert mask to RLE
def mask_to_rle(binary_mask: np.ndarray) -> str:
    mask_fortran = np.asfortranarray(binary_mask)
    rle_dict = mask_utils.encode(mask_fortran)
    if isinstance(rle_dict["counts"], bytes):
        rle_dict["counts"] = rle_dict["counts"].decode("utf-8")
    return json.dumps(rle_dict)

    # Convert an RLE-format string to a mask


def rle_to_mask(rle_str: str) -> np.ndarray:
    # The CSV stores a Python dict string (single quotes); parse it with ast.literal_eval
    rle_dict = ast.literal_eval(rle_str)

    decoded_mask = mask_utils.decode(rle_dict)
    return decoded_mask


import cv2
import numpy as np


def auto_crop_receipt(img, area_threshold=0.7, padding=0.01):
    """
    Automatically detect and crop the core region of a receipt
    :param image_path: image path
    :param area_threshold: area-ratio threshold that triggers cropping (e.g. 0.6 means cropping is triggered when the receipt content occupies less than 60%)
    :param padding: pixel margin retained around the crop to prevent text from being cut off
    :return: cropped image (or original image), crop coordinates (x, y, w, h), whether cropping was performed (Boolean)
    """
    # # 1. Read the image
    # img = cv2.imread(image_path)

    # if img is None:
    #     raise ValueError("Image read failed, please check the path")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    total_area = original_h * original_w
    padding = int(padding * max(original_w, original_h))
    # print("padding",padding)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Binarization (invert colors: make text white and background black to ease contour finding)
    # Use OTSU to automatically find the optimal threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Morphological dilation (key step!)
    # Define a large rectangular kernel to merge adjacent text, lines, and paragraphs into a single block
    # The kernel size should be fine-tuned for your image resolution; (50, 50) suits high-resolution images
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(0.04 * original_w), int(0.05 * original_h))
    )
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 5. Find contours
    # RETR_EXTERNAL only retrieves the outermost contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours",len(contours))
    if not contours or len(contours) < 5:
        return img, (0, 0, original_w, original_h), False, (0, 0, 0, 0)

    # 6. Find the largest contour by area (i.e. the receipt body)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 7. Decide whether to crop
    receipt_area = w * h
    coverage_ratio = receipt_area / total_area
    if (
        coverage_ratio < 0.05
    ):  # If after cropping less than 0.3 of the original remains, this may be a detection error, so do not crop
        # print("Insufficient crop",coverage_ratio)
        return img, (0, 0, original_w, original_h), False, (0, 0, 0, 0)
    if coverage_ratio < area_threshold:
        # Cropping is needed; add padding and prevent out-of-bounds
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(original_w, x + w + padding)
        y_end = min(original_h, y + h + padding)

        cropped_img = img[y_start:y_end, x_start:x_end]
        crop_box = (x_start, y_start, x_end, y_end)

        return (
            cropped_img,
            crop_box,
            True,
            (x_start, original_w - x_end, y_start, original_h - y_end),
        )
    else:
        # Content occupies enough area; no cropping needed
        return img, (0, 0, original_w, original_h), False, (0, 0, 0, 0)


train_newd_crop = [
    "f6afd4b3787747bca22f778764b28199",
    "0ff57252ea5c481f86113ef8eea4228c",
    "b7609bade39b4458b8df71c5281b74cb",
    "feec59b9fc6e42c8a855bc76040a30cd" "3fa1bc0a97654113963be1bde63a36ba",
    "7e6f1a518f7142019dcd121677a75764",
    "28e04ba45b364237ac1c719ac96e2bd5",
    "30ff90438f2f4bf5a4fef343aa04021f",
    "ca380289c7704e61968f9620626b33d1",
    "31e02f7f6de040f2aa9f5d8d9a51ae80",
    "8840992135d54909a810e204a2f61ae4",
]
test_newd_crop = [
    "1d8608b2c74d4b6999b5eb1188465e99",
    "3c9b165981fc43eeb58988f3fdefc66f",
    "3e22a4e0f6054029b64a2e339618c96d",
    "9fd439c4a3044d6c8b2e1de9e929bd23",
    "10ca0ebc864541179d2c789672ddba7b",
    "88ee8eea42b94a7397ae296159e0b699",
    "895e5c06e3004f5a9a9938cb16c6f3f2",
    "1763cebb85f74f3c8202af4fcfe57f4a",
    "08599d271df94bb7a59c328df2cca189",
    "64047e7f9dbf4faeaeb6eef2d57ce544",
    "001858037f7846a79c619fda3d915e75",
    "2641942bbefb4f27a80669ccdee3d459",
    "04630109324c42f99f75750324a4bf1b",
    "a09d96691bbe4788806f192b1ece1c4a",
    "a403b73740ec4c5ba720b66188c8a179",
    "acefacb475284784a1a7ac08be375d97",
    "ad55d3ce502c436ca9edc71da3fe371b",
    "b8ae442fda8f40dd8265cbd6c715acc0",
    "bcd92b64411e4fab91b240a4aab3bcb3",
    "cde2ae3d27144afca269036dc3873030",
    "d09f338142794956a9874ed399163a56",
    "d024bfb0ca1b4f02a1fa23691639a335",
    "de2ddf55aa2743c88b5bda4dec138e06",
    "e74395e435df45c7ae8a6d40abdbde30",
    "de6033da5e3e42429dd3be156e6ce18b",
]


#  88ee8eea42b94a7397ae296159e0b699.jpg acefacb475284784a1a7ac08be375d97.jpg cropping does not work well
class IdbmDataset(data.Dataset):
    """
    Predict whether an image is forged, and output the corresponding pixel-level mask
    """

    def __init__(self, data_list, config=None, mode="train"):
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
        self.rel_num = len(self.data_list)

        self.transform = self.init_data_aug_method()
        # self.common_transform, self.crop_transform, self.resize_transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose(
            [
                A.HorizontalFlip(p=self.config["data_aug"]["flip_prob"]),
                A.VerticalFlip(p=self.config["data_aug"]["flip_prob"]),
                A.PadIfNeeded(  # this ensures small images are unchanged
                    min_height=self.config["resolution"],
                    min_width=self.config["resolution"],
                    border_mode=cv2.BORDER_CONSTANT,  # use a constant border
                    value=0,
                    mask_value=0,
                ),
                A.Rotate(
                    limit=self.config["data_aug"]["rotate_limit"],
                    p=self.config["data_aug"]["rotate_prob"],
                ),
                A.GaussNoise(p=self.config["data_aug"]["GaussNoise"]),
                A.GaussianBlur(
                    blur_limit=self.config["data_aug"]["blur_limit"],
                    p=self.config["data_aug"]["blur_prob"],
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=96,
                    max_width=96,
                    min_holes=1,
                    min_height=32,
                    min_width=32,
                    fill_value=0,
                    p=self.config["data_aug"]["CoarseDropout_prob"],
                ),
                # saturation
                # A.HueSaturationValue(hue_shift_limit=self.config['data_aug']['hsl'],
                #                       sat_shift_limit=self.config['data_aug']['ssl'],
                #                         val_shift_limit=self.config['data_aug']['vsl'],
                #                           p=self.config['data_aug']['hst_prob']),
                # RGB three-channel principal component perturbation
                # A.FancyPCA(alpha=self.config['data_aug']['fpca_alpha'], p=self.config['data_aug']['fpca_prob']),
                # A.RandomScale(scale_limit=(-0.5, 0), p=0.3),  # shrink the image to 50%-100% of its original size
                # A.PadIfNeeded(
                #         min_height=self.config['resolution'],
                #         min_width=self.config['resolution'],
                #         border_mode=cv2.BORDER_REPLICATE,  # stretch the border (or use BORDER_WRAP)
                #         p=1
                #             ),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config["data_aug"]["brightness_limit"],
                    contrast_limit=self.config["data_aug"]["contrast_limit"],
                    p=self.config["data_aug"]["brightness_prob"],
                ),
                A.FancyPCA(
                    alpha=self.config["data_aug"]["fpca_alpha"],
                    p=self.config["data_aug"]["fpca_prob"],
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.config["data_aug"]["hsl"],
                    sat_shift_limit=self.config["data_aug"]["ssl"],
                    val_shift_limit=self.config["data_aug"]["vsl"],
                    p=self.config["data_aug"]["hst_prob"],
                ),
                # A.OneOf([A.CoarseDropout(max_width=16, max_height=16,p=1)], p=self.config['data_aug']['GridDropout']),
                # A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.1),
                A.ToGray(p=self.config["data_aug"]["ToGray"]),
                # A.OneOf([A.CoarseDropout(max_width=16, max_height=16,p=1)], p=self.config['data_aug']['GridDropout']),
                A.Resize(
                    height=self.config["resolution"],
                    width=self.config["resolution"],
                    p=1,
                ),  # only resize images that exceed the resolution
                # 1. Keep the aspect ratio and scale the longest side to size
                # A.LongestMaxSize(max_size=self.config['resolution'], interpolation=cv2.INTER_AREA),
                # 2. Pad the short side to size * size
                # border_mode=cv2.BORDER_CONSTANT means fill with a constant solid color
                # value is the image fill color (usually 0, i.e. black)
                # mask_value is the mask fill color (usually 0, i.e. background)
                # A.PadIfNeeded( min_height=self.config['resolution'], min_width=self.config['resolution'],
                #                     border_mode=cv2.BORDER_CONSTANT,  value=0, mask_value=0)
                #    A.RandomCrop(height=self.config['resolution'], width=self.config['resolution'], p=1.0)
                # A.OneOf([A.Resize(height=self.config['resolution'], width=self.config['resolution'], p=1),
                #  A.RandomCrop(height=self.config['resolution'], width=self.config['resolution'], p=1.0)], p=1.0)
            ],
        )
        return trans

    # def init_data_aug_method(self):
    #     size = self.config['resolution']

    #     # 1. Pixel-level and generic augmentation (does not change image size)
    #     # Remove the original A.Resize from here
    #     common_trans = A.Compose([
    #         A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
    #         A.VerticalFlip(p=self.config['data_aug']['flip_prob']),
    #         A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
    #         A.GaussNoise(p=self.config['data_aug']['GaussNoise']),
    #         A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
    #         A.CoarseDropout(
    #             max_holes=8, max_height=96, max_width=96,
    #             min_holes=1, min_height=32, min_width=32,
    #             fill_value=0, p=self.config['data_aug']['CoarseDropout_prob']
    #         ),
    #         A.RandomBrightnessContrast(
    #             brightness_limit=self.config['data_aug']['brightness_limit'],
    #             contrast_limit=self.config['data_aug']['contrast_limit'],
    #             p=self.config['data_aug']['brightness_prob']
    #         ),
    #         A.FancyPCA(alpha=self.config['data_aug']['fpca_alpha'], p=self.config['data_aug']['fpca_prob']),
    #         A.HueSaturationValue(
    #             hue_shift_limit=self.config['data_aug']['hsl'],
    #             sat_shift_limit=self.config['data_aug']['ssl'],
    #             val_shift_limit=self.config['data_aug']['vsl'],
    #             p=self.config['data_aug']['hst_prob']
    #         ),
    #         A.ToGray(p=self.config['data_aug']['ToGray'])
    #     ])

    #     # 2. Spatial transform for extreme aspect ratios: pad + random crop
    #     crop_trans = A.Compose([
    #         A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    #         A.RandomCrop(height=size, width=size, p=1.0)
    #     ])

    #     # 3. Spatial transform for normal aspect ratios: direct Resize
    #     resize_trans = A.Compose([
    #         A.Resize(height=size, width=size, p=1.0)
    #     ])

    #     return common_trans, crop_trans, resize_trans

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

        return np.array(img, dtype=np.uint8)

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

    def data_aug(self, img, augmentation_seed=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        # Create a dictionary of arguments
        kwargs = {"image": img}
        if mask is not None:
            kwargs["mask"] = mask

        # Check if the landmark and mask are not None

        # Apply data augmentation
        # h, w = img.shape[:2]
        # aspect_ratio = max(h, w) / min(h, w)
        # if aspect_ratio > 1.8:
        #     spatial_transformed = self.crop_transform(**kwargs)
        # else:
        #     spatial_transformed = self.resize_transform(**kwargs)
        # transformed = self.common_transform(**spatial_transformed)
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed["image"]
        if mask is not None:
            augmented_mask = transformed["mask"]

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()
        if mask is not None:
            return augmented_img, augmented_mask
        return augmented_img

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
        data = self.data_list[index]
        # Load the image
        is_crop = False
        padding = [0, 0, 0, 0]
        image = self.load_rgb(data["image_path"])
        if image.shape[0] > 7000 or (
            data.get("forgery", 0) and data["image_path"].name[:-4] in train_newd_crop
        ):
            image, crop_box, is_crop, padding = auto_crop_receipt(image)

        label = data["label"]
        ## Load the corresponding mask
        if label == 1:
            if data.get("extend", 0):
                gt_mask = rle_to_mask(data["mask_rel"])
                gt_mask = gt_mask * 255
            else:
                gt_mask = cv2.imread(data["mask_path"], cv2.IMREAD_GRAYSCALE)
            if is_crop and gt_mask.shape != image.shape[:-1]:
                ## If cropped, the mask must be cropped too
                # print("Cropping image:", data['image_path'], "crop box:", image.shape, "original mask shape:", gt_mask.shape)
                gt_mask = gt_mask[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            # mask_rle = data['rel_mask']
            # gt_mask = rle_to_mask(mask_rle)
        else:
            gt_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # print("After cropping",gt_mask.shape)
        # # If sizes differ, force resize (use nearest-neighbor interpolation to preserve the 0-1 property)
        if gt_mask.shape != image.shape[:-1]:
            ## For some external images, the original mask is not aligned with the image; resize directly in this case
            gt_mask = cv2.resize(
                gt_mask, image.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST
            )
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        # Do Data Augmentation
        if self.mode == "train" and self.config["use_data_augmentation"]:
            image, gt_mask = self.data_aug(image, mask=gt_mask)
        if self.mode == "test":
            # image = isotropically_resize_and_pad(image,size=size )
            # gt_mask = isotropically_resize_and_pad(gt_mask,size=size )
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            gt_mask = cv2.resize(gt_mask, (size, size), interpolation=cv2.INTER_CUBIC)

        image_trans = self.normalize(self.to_tensor(image))
        gt_mask = (gt_mask > 128).astype(np.float32)
        gt_mask = torch.from_numpy(gt_mask).float()
        gt_mask = torch.unsqueeze(gt_mask, 0)
        ## Augmentation may have changed the label
        if gt_mask.any():
            label = 1
        else:
            label = 0
        label = torch.tensor(label).float()

        data_dict = {
            "image_path": str(data["image_path"]),
            "image": image_trans,
            "gt_mask": gt_mask,
            "label": label,  # forged/real
            # 'padding':padding
        }
        return data_dict

    def __len__(self):

        return self.lenght


if __name__ == "__main__":
    with open(
        "<BASE_PATH>/2025_CSIRO/baseline/training/config/detector/biom_vit.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)
    data_pl = pl.read_csv("<BASE_PATH>/2025_CSIRO/train_data.csv")
    data_list = [
        (Path(config["rgb_dir"]) / path, label)
        for path, label in zip(
            data_pl["image_path"].to_list(),
            data_pl.select(config["target_columns"]).to_numpy(),
        )
    ]
    data_list = np.array(data_list, dtype=object)
    train_set = BiomDataset(
        data_list=data_list,
        config=config,
        mode="train",
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config["train_batchSize"],
        shuffle=True,
        num_workers=0,
    )
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(batch)
        break
