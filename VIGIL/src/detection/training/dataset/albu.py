import random

import cv2
import numpy as np
import torch
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops.functional import crop


class DeNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 1, -1)
        self.std = torch.tensor(std).view(1, 1, -1)

    def __call__(self, tensor):
        # input is w,h,c
        tensor = tensor * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)  # ensure the range is within 0~1
        tensor = (tensor.numpy() * 255).astype(np.uint8)
        return tensor


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


def isotropically_resize_image(
    img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC
):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(
        self,
        max_side,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        always_apply=False,
        p=1,
    ):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(
        self,
        img,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        **params
    ):
        return isotropically_resize_image(
            img,
            size=self.max_side,
            interpolation_down=interpolation_down,
            interpolation_up=interpolation_up,
        )

    def apply_to_mask(self, img, **params):
        return self.apply(
            img,
            interpolation_down=cv2.INTER_NEAREST,
            interpolation_up=cv2.INTER_NEAREST,
            **params
        )

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(
            img,
            (w, h),
            interpolation=random.choice(
                [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]
            ),
        )
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(
            mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1])
        )
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"
