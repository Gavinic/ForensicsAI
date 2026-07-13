import cv2
import numpy as np


def gaussian_blur(image, ksizes=[3, 5, 7, 9, 11, 13, 15], p=0.5):
    if np.random.rand() <= p:
        if ksizes is None:
            k = np.random.choice([3, 5, 7, 9, 11, 13, 15])
        else:
            k = np.random.choice(ksizes)

        gb_image = cv2.GaussianBlur(image, ksize=(k, k), sigmaX=k * 1.0 / 6)
        return gb_image
    return image
