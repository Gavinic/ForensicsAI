import cv2
import numpy as np


def rotate(image, mask, p=0.5):
    if np.random.rand() < p:
        i = np.random.randint(0, 3)
        if i == 0:
            rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotate_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif i == 1:
            rotate_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotate_mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotate_image = cv2.rotate(image, cv2.ROTATE_180)
            rotate_mask = cv2.rotate(mask, cv2.ROTATE_180)
        return rotate_image, rotate_mask
    return image, mask
