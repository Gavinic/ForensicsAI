import numpy as np


def pad_image(image, pad_height, pad_width):
    if len(image.shape) == 3:
        return np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return np.pad(
        image, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0
    )


def crop(images, masks, crop_size=(256, 256), mode="center"):
    # padding
    height, width = images[0].shape[:2]
    pad_height = crop_size[0] - height if height < crop_size[0] else 0
    pad_width = crop_size[1] - width if width < crop_size[1] else 0
    height, width = height + pad_height, width + pad_width

    # crop
    if mode == "center":
        center = (height // 2, width // 2)
    elif mode == "random":
        center = (
            np.random.randint(crop_size[0] // 2, height + 1 - crop_size[0] // 2),
            np.random.randint(crop_size[1] // 2, width + 1 - crop_size[1] // 2),
        )
    lh, rh = center[0] - crop_size[0] // 2, center[0] + crop_size[0] // 2
    lw, rw = center[1] - crop_size[1] // 2, center[1] + crop_size[1] // 2

    crop_images, crop_masks = [], []
    for image in images:
        crop_images.append(pad_image(image, pad_height, pad_width)[lh:rh, lw:rw])
    for mask in masks:
        crop_masks.append(pad_image(mask, pad_height, pad_width)[lh:rh, lw:rw])
    return crop_images, crop_masks


if __name__ == "__main__":
    a = np.random.random((541, 523, 3))
    b = np.random.random((541, 523, 3))
    c = np.random.random((541, 523))
    [a1, b1], [c1] = crop([a, b], [c], crop_size=(256, 256), mode="random")
    print(a1.shape, b1.shape, c1.shape)
