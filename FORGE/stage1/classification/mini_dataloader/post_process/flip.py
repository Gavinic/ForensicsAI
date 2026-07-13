import cv2
import numpy as np


def horizontal_flip(image, mask, p=0.5):
    if np.random.rand() < p:
        hflip_image = cv2.flip(image, 1)
        hflip_mask = cv2.flip(mask, 1)
        return hflip_image, hflip_mask
    return image, mask


def vertical_flip(image, mask, p=0.5):
    if np.random.rand() < p:
        vflip_image = cv2.flip(image, 0)
        vflip_mask = cv2.flip(mask, 1)
        return vflip_image, vflip_mask
    return image, mask
