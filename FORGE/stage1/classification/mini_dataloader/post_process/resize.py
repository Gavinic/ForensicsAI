from typing import Tuple

import cv2
import numpy as np


def resize(image, mask, rates=[0.78], p=0.5):
    if np.random.rand() <= p:
        if rates is None:
            rate = np.random.choice(np.arange(25, 400, 1)) / 100.0
        else:
            rate = np.random.choice(rates)

        ori_size = image.shape[:2]
        resize_image = cv2.resize(
            image, (int(ori_size[1] * rate), int(ori_size[0] * rate)), cv2.INTER_AREA
        )
        resize_mask = cv2.resize(
            mask, (int(ori_size[1] * rate), int(ori_size[0] * rate)), cv2.INTER_NEAREST
        )

        return resize_image, resize_mask
    return image, mask


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (neww, newh)


def resize_image_longest_side(image: np.ndarray, target_length: int) -> np.ndarray:
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    return cv2.resize(image, target_size, cv2.INTER_AREA)


def resize_mask_longest_side(mask: np.ndarray, target_length: int) -> np.ndarray:
    """
    Expects a numpy array with shape HxW in uint8 format.
    """
    target_size = get_preprocess_shape(mask.shape[0], mask.shape[1], target_length)
    return cv2.resize(mask, target_size, cv2.INTER_NEAREST)
