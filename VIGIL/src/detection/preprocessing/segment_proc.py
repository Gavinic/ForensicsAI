import glob

# from joblib import Parallel, delayed
import json
import logging
import os
import pathlib
import pickle
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import segmentation_models_pytorch as smp
import torch
import xgboost as xgb
from PIL import Image, ImageCms
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from skimage import color, io
from tqdm import tqdm


class VegAnnModel(torch.nn.Module):
    def __init__(
        self, arch: str, encoder_name: str, in_channels: int, out_classes: int, **kwargs
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_outputs, self.val_outputs, self.test_outputs = [], [], []

    def forward(self, image: torch.Tensor):
        # normalize image here #todo
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


def colorTransform_VegGround(
    im: np.ndarray, X_true: np.ndarray, alpha_vert: float, alpha_g: float
) -> np.ndarray:
    """Add color overlay for vegetation and ground"""
    image = np.copy(im)

    # Ground color (brown)
    ground_color = np.array([97, 65, 38])
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 0,
            image[:, :, c] * (1 - alpha_vert) + alpha_vert * ground_color[c],
            image[:, :, c],
        )

    # Vegetation color (green)
    veg_color = np.array([34, 139, 34])
    for c in range(3):
        image[:, :, c] = np.where(
            X_true == 1,
            image[:, :, c] * (1 - alpha_g) + alpha_g * veg_color[c],
            image[:, :, c],
        )
    return image


def get_features(image: np.ndarray) -> np.ndarray:
    """Extract multiple color-space features"""
    pil_image = Image.fromarray(image)

    # Convert to different color spaces
    hsv = np.array(pil_image.convert(mode="HSV"))

    srgb_p = ImageCms.createProfile("sRGB")
    lab_p = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(pil_image, rgb2lab))

    ycbcr = np.array(pil_image.convert(mode="YCbCr"))
    Labb = color.rgb2lab(image)

    # RGB channels
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # HSV features
    h = (hsv[:, :, 0] * 360) / 255
    s = hsv[:, :, 1] / 2.55

    # Lab features
    a = Labb[:, :, 1]
    bb = Lab[:, :, 2]

    # Grayscale
    ge = np.mean([r, g, b], axis=0)

    # CMY features
    CMYlist = np.array(
        [np.min(idx) for idx in zip(1 - r / 255, 1 - g / 255, 1 - b / 255)]
    )
    m = ((1 - g / 255 - CMYlist) / (1 - CMYlist)) * 100
    ye = ((1 - b / 255 - CMYlist) / (1 - CMYlist)) * 100

    # YCbCr features
    cb, cr = ycbcr[:, :, 1], ycbcr[:, :, 2]

    # IQ features
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    # Stack all features
    model_input = np.stack(
        (r, g, b, h, s, a, bb, ge, m, ye, cb, cr, i, q), axis=2
    ).squeeze(0)
    return np.nan_to_num(model_input)


def automatic_contrast(image: np.ndarray, clip: float = 0.01):
    """Automatic contrast adjustment"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Compute cumulative distribution
    cumul = np.cumsum(hist.flatten())

    # Compute clipping point
    maximum = cumul[-1]
    clip_value = (maximum / 100.0) * clip / 2.0

    # Find the minimum and maximum grayscale values
    minimum_gray = np.argmax(cumul >= clip_value)
    maximum_gray = np.argmax(cumul >= (maximum - clip_value))

    # Compute alpha and beta
    alpha = (255 / (maximum_gray - minimum_gray)) + 0.4
    beta = (-minimum_gray * alpha) * 2

    contr_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contr_result, alpha, beta


def prediction_XG_SVM(image, model, threshold, contrasted, mask):

    # Ff contrasted apply image contrast
    if contrasted == 1:
        image, _, _ = automatic_contrast(image)

    # Revert channels to have RGB /!\ IMPORTANT
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # If Mask doesn't exist create one full of 1's
    if mask is None:
        mask = np.ones((height, width))

    # Flatten arrays
    image = image.reshape((width * height), 1, 3)
    mask = mask.reshape((width * height), 1)

    # Select indices of vegetation pixels
    yellow_green_mask = np.zeros(image.shape[:-1])  # Null array to set bckg pixels to 0
    vegetation_pixels = mask > 0  # Get vegetation pixels indices
    image = image[None, vegetation_pixels]

    # Apply preprocessing to add features to each pixels
    featured_image = get_features(image)

    # Predictions on vegetation pixels
    # y_pred = (model.predict_proba(featured_image)[:,1] >= threshold).astype(int)
    y_pred = model.predict_proba(featured_image)[:, 1]

    # Replace pixels of null array at vegetation indices with output of classification
    yellow_green_mask[vegetation_pixels] = y_pred
    yellow_green_mask = yellow_green_mask.reshape((height, width))

    # Reshape
    # mask[(mask == 1) & (yellow_green_mask != 1)] = 2
    # mask = mask.reshape((height, width))
    # Rotate image
    # print(mask.shape)
    # print(yellow_green_mask.shape)

    return yellow_green_mask  # mask


# Function from https://github.com/and-jonas/wheat-segmentation-models
def get_features_Necrosis(image: np.ndarray) -> np.ndarray:

    img_RGB = np.array(image / 255, dtype=np.float32)
    img_RGB = img_RGB[:, :, :3]

    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    lambda_r, lambda_g, lambda_b = 670, 550, 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    descriptors = np.concatenate(
        [
            img_RGB,
            img_HSV,
            img_Lab,
            img_Luv,
            img_YUV,
            img_YCbCr,
            np.stack([ExG, ExR, TGI], axis=2),
        ],
        axis=2,
    )
    descriptor_names = [
        "sR",
        "sG",
        "sB",
        "H",
        "S",
        "V",
        "L",
        "a",
        "b",
        "L",
        "u",
        "v",
        "Y",
        "U",
        "V",
        "Y",
        "Cb",
        "Cr",
        "ExG",
        "ExR",
        "TGI",
    ]
    return descriptors


def classify_Necrosis(
    image: np.ndarray,
    model,
    contrasted: int = 0,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Be sure image is in BGR ! (to apply correctly contrast if needed)
    if binary veg/bckg mask is given: apply model only on vegetation pixels
    """
    # Reverse channels to have RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # If no mask, create one full of 1
    if mask is None:
        mask = np.ones((height, width))
    # If mask is given and there is no vegetation in mask: returns mask with only zeros
    if 1 not in np.unique(mask):
        return np.zeros(mask.shape)

    image = image.reshape((width * height), 1, 3)
    mask = mask.reshape((width * height), 1)
    yellow_green_mask = np.zeros(image.shape[:-1])
    vegetation_pixels = mask > 0
    image = image[None, vegetation_pixels]

    try:
        featured_image = get_features_Necrosis(image)
    except:
        # If full of Soil
        mask = np.zeros((height, width), dtype=np.uint8)
        return mask

    descriptors_flatten = featured_image.reshape(-1, featured_image.shape[-1])
    descriptors_flatten = xgb.DMatrix(descriptors_flatten)
    segmented_flatten_probs = model.predict(descriptors_flatten)
    y_pred = np.argmax(segmented_flatten_probs, axis=1)  # J0 V1 M2
    y_pred = [
        3 if x == 0 else x for x in y_pred
    ]  # Move J0 to J3 to let Soil as 0 ==> Final : 0 Soil | 1 Healthy | 2 Necrosis-Brown | 3 Chlorosis-Yellow

    yellow_green_mask[vegetation_pixels] = y_pred
    mask = yellow_green_mask.reshape((height, width))

    return mask


def visualisation(rgb_image: np.ndarray, yg_mask: np.ndarray) -> np.ndarray:

    image_copy = rgb_image.copy()
    image_copy[yg_mask == 1] = (0, 100, 0)
    image_copy[yg_mask == 2] = (0, 215, 255)

    visualisation = cv2.addWeighted(rgb_image, 0.25, image_copy, 0.75, 0)

    return visualisation


def load_vegannmodel(ckt_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckt_path, map_location=device)
    model = VegAnnModel("Unet", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)  # Move the model to the selected device
    # preprocess_fn = smp.encoders.get_preprocessing_fn("resnet34", pretrained="imagenet")
    preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")
    model.eval()
    return model, preprocess_input


# def predict_mask(image, dl_model, preprocess_input, device):
#     image = cv2.imread(imname)

if __name__ == "__main__":
    ## Load the DL model
    ckt_path = "<BASE_PATH>/2025_CSIRO/VegAnnUnet_rest34.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess_input = load_vegannmodel(ckt_path, device)
    ## Load the XGBoost model
    xgb_model_path = "<BASE_PATH>/2025_CSIRO/XGBoost"
    model_XG = pickle.load(Path(xgb_model_path).open("rb"))

    svm_model_path = "<BASE_PATH>/2025_CSIRO/model_scikit"
    model_SVM = pickle.load(Path(svm_model_path).open("rb"))

    new_attrs = [
        "grow_policy",
        "max_bin",
        "eval_metric",
        "callbacks",
        "early_stopping_rounds",
        "max_cat_to_onehot",
        "max_leaves",
        "sampling_method",
        "enable_categorical",
        "feature_types",
        "max_cat_threshold",
        "predictor",
    ]
    for attr in new_attrs:
        setattr(model_XG, attr, None)
    ## Inference
    for imname in tqdm(glob.glob("<BASE_PATH>/2025_CSIRO/data/train/*.jpg")):
        image = cv2.imread(imname)
        image = cv2.resize(image, (2048, 1024))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = rgb_image.shape[1], rgb_image.shape[0]
        md = w // 2
        image_l = rgb_image[:, :md, :]
        image_r = rgb_image[:, md:, :]

        image_l = preprocess_input(image_l).astype("float32")
        image_r = preprocess_input(image_r).astype("float32")
        image_l = torch.tensor(image_l).permute(2, 0, 1)  # , dtype=float
        image_r = torch.tensor(image_r).permute(2, 0, 1)  # , dtype=float
        inputs = torch.stack((image_l, image_r), dim=0)
        # print(inputs.size)
        # inputs = inputs
        inputs = inputs.to(device)  # Move input tensor to the selected device
        # print(inputs.shape)
        with torch.no_grad():
            logits = model(inputs)
        pr_mask = logits.sigmoid().cpu().numpy()
        pr_mask = np.concatenate((pr_mask[0, 0], pr_mask[1, 0]), axis=-1)
        pred = (pr_mask > 0.55).astype(np.uint8)
        raw_yellow_green_mask = prediction_XG_SVM(
            image, model_XG, mask=pred, threshold=0.5, contrasted=1
        )

        result = np.stack([pr_mask, raw_yellow_green_mask], axis=2)
        result = cv2.resize(result, (2000, 1000))  # restore to original size
        np.save(
            os.path.join(
                "<BASE_PATH>/2025_CSIRO/data_mask",
                os.path.basename(imname).replace(".jpg", ".npy"),
            ),
            result,
        )

        # yellow_green_mask = cv2.erode(raw_yellow_green_mask, np.ones((2,2), np.uint8), iterations=1)
        # yellow_green_mask = cv2.GaussianBlur(yellow_green_mask, (3,3), 0)
