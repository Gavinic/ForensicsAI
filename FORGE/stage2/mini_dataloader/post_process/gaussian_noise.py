import numpy as np


def gaussian_noise(image, mean=0, sds=[3], p=0.5):
    if np.random.rand() <= p:
        if sds is None:
            sd = np.random.choice([3, 5, 7, 9, 11, 13])
        else:
            sd = np.random.choice(sds)
        noise = np.random.normal(mean, sd, image.shape)

        gn_image = np.clip(image + noise, 0, 255)
        return gn_image
    return image
