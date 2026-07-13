import cv2
import numpy as np


def jpeg_compression(image, qfs=[75], p=0.5):
    if np.random.rand() <= p:
        if qfs is None:
            qf = np.random.choice(np.arange(50, 95, 1))
        else:
            qf = np.random.choice(qfs)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]

        result, encimg = cv2.imencode(".jpg", image, encode_param)
        jc_image = cv2.imdecode(encimg, 1)
        return jc_image
    return image
