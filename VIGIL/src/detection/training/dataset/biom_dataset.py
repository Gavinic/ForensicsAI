# description: Abstract Base Class for all types of deepfake datasets.

import sys

sys.path.append(".")

import glob
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
from torch.utils import data
from torchvision import transforms as T

from .albu import IsotropicResize


class BiomDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
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

    def init_data_aug_method(self):
        trans = A.Compose(
            [
                A.HorizontalFlip(p=self.config["data_aug"]["flip_prob"]),
                A.VerticalFlip(p=self.config["data_aug"]["flip_prob"]),
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
                A.Resize(
                    height=self.config["resolution"],
                    width=self.config["resolution"],
                    p=1,
                ),
            ],
        )
        return trans

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

    def block_swap_augmentation(self, left, right, num_blocks=4, swap_prob=0.5):
        """
        Quickly swap blocks of the left and right images using tensor operations

        Args:
            left: left image (H, W, C)
            right: right image (H, W, C)
            num_blocks: number of blocks each image is split into (default 4, i.e. 2x2)
            swap_prob: probability of swapping a block

        Returns:
            left, right: images after swapping
        """
        h, w, c = left.shape
        rows = int(np.sqrt(num_blocks))
        cols = int(np.sqrt(num_blocks))

        block_h = h // rows
        block_w = w // cols

        # Reshape to (rows, block_h, cols, block_w, c)
        left_blocks = left.reshape(rows, block_h, cols, block_w, c)
        right_blocks = right.reshape(rows, block_h, cols, block_w, c)

        # Transpose to (rows, cols, block_h, block_w, c) for easier manipulation
        left_blocks = left_blocks.transpose(0, 2, 1, 3, 4)
        right_blocks = right_blocks.transpose(0, 2, 1, 3, 4)

        # Generate random swap mask: (rows, cols) boolean array
        swap_mask = np.random.random((rows, cols)) < swap_prob

        # Expand mask dimensions to match block dimensions (rows, cols, 1, 1, 1)
        swap_mask_expanded = swap_mask[:, :, None, None, None]

        # Use numpy's where for conditional swapping
        left_result = np.where(swap_mask_expanded, right_blocks, left_blocks)
        right_result = np.where(swap_mask_expanded, left_blocks, right_blocks)

        # Transpose back to (rows, block_h, cols, block_w, c)
        left_result = left_result.transpose(0, 2, 1, 3, 4)
        right_result = right_result.transpose(0, 2, 1, 3, 4)

        # Reshape back to the original shape
        left_aug = left_result.reshape(h, w, c)
        right_aug = right_result.reshape(h, w, c)

        return left_aug, right_aug

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
        # index = index % self.rel_num
        data = self.data_list[index]
        # augmentation_seed = random.randint(0,2**11)
        # Load the image
        image = self.load_rgb(data["image_path"])
        ## Load the corresponding mask
        # mask = np.load(os.path.join("<BASE_PATH>/2025_CSIRO/data_mask",
        #                             os.path.basename(data['image_path']).replace('.jpg','.npy')))
        # image = np.array(image)  # Convert to numpy array for data augmentation

        h, w, _ = image.shape
        mid = w // 2
        if "extend" in data:
            left = image
            right = image.copy()
        else:
            left = image[:, :mid]
            right = image[:, mid:]
        # mask_left = mask[:, :mid]
        # mask_right = mask[:, mid:]
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        # Do Data Augmentation
        if self.mode == "train" and self.config["use_data_augmentation"]:
            left = self.data_aug(left)
            right = self.data_aug(right)
            # left, right = self.block_swap_augmentation(left, right,num_blocks=4)
        if self.mode == "test":
            left = cv2.resize(left, (size, size), interpolation=cv2.INTER_CUBIC)
            right = cv2.resize(right, (size, size), interpolation=cv2.INTER_CUBIC)
            # mask_left = cv2.resize(mask_left, (size, size), interpolation=cv2.INTER_CUBIC)
            # mask_right = cv2.resize(mask_right, (size, size), interpolation=cv2.INTER_CUBIC)

        left_trans = self.normalize(self.to_tensor(left))
        right_trans = self.normalize(self.to_tensor(right))
        # mask_left = torch.tensor(mask_left, dtype=torch.float32) # H,W,2
        # mask_right = torch.tensor(mask_right, dtype=torch.float32)
        # label =  np.log1p(label)

        data_dict = {
            "image_path": str(data["image_path"]),
            "image_left": left_trans,
            "image_right": right_trans,
            # "mask_left":mask_left,
            # "mask_right":mask_right,
        }
        for name in self.config["target_columns"]:
            data_dict[name] = torch.tensor([data[name]], dtype=torch.float32)

        ## Add two more

        data_dict["Height_Ave_cm"] = torch.tensor(
            [data["Height_Ave_cm"]], dtype=torch.float32
        )

        data_dict["Pre_GSHH_NDVI"] = torch.tensor(
            [data["Pre_GSHH_NDVI"]], dtype=torch.float32
        )

        data_dict["State"] = torch.tensor(data["State"], dtype=torch.long)
        data_dict["Species"] = torch.tensor(data["Species"], dtype=torch.float32)
        return data_dict

    def __len__(self):

        return self.lenght


class BiomPretrainDataset(BiomDataset):
    """
    Abstract base class for all deepfake datasets.
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

    def init_data_aug_method(self):
        trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=[-15, 15], p=0.15),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=self.config["data_aug"]["blur_limit"], p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=64,
                    max_width=64,
                    min_holes=1,
                    min_height=32,
                    min_width=32,
                    fill_value=0,
                    p=0.3,
                ),
                # saturation
                A.HueSaturationValue(
                    hue_shift_limit=self.config["data_aug"]["hsl"],
                    sat_shift_limit=self.config["data_aug"]["ssl"],
                    val_shift_limit=self.config["data_aug"]["vsl"],
                    p=0.3,
                ),
                # RGB three-channel principal component perturbation
                A.FancyPCA(alpha=self.config["data_aug"]["fpca_alpha"], p=0.3),
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
                    p=0.3,
                ),
                # A.OneOf([A.CoarseDropout(max_width=16, max_height=16,p=1)], p=self.config['data_aug']['GridDropout']),
                A.ImageCompression(
                    quality_lower=self.config["data_aug"]["quality_lower"],
                    quality_upper=self.config["data_aug"]["quality_upper"],
                    p=0.1,
                ),
                A.ToGray(p=0.15),
                A.Resize(
                    height=self.config["resolution"],
                    width=self.config["resolution"],
                    p=1,
                ),
            ],
        )
        return trans

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
        # index = index % self.rel_num
        img_path = self.data_list[index]

        augmentation_seed = None
        # Load the image
        try:
            image = self.load_rgb(img_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        # image = np.array(image)  # Convert to numpy array for data augmentation
        left = image
        right = image.copy()

        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        # Do Data Augmentation
        if self.mode == "train" and self.config["use_data_augmentation"]:
            left = self.data_aug(left, augmentation_seed)
            right = self.data_aug(right, augmentation_seed)
        if self.mode == "test":
            left = cv2.resize(left, (size, size), interpolation=cv2.INTER_CUBIC)
            right = cv2.resize(right, (size, size), interpolation=cv2.INTER_CUBIC)

        left_trans = self.normalize(self.to_tensor(left))
        right_trans = self.normalize(self.to_tensor(right))

        data_dict = {
            "image_path": img_path,
            "image_left": left_trans,
            "image_right": right_trans,
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
