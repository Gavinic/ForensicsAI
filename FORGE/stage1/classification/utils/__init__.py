import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from .auc import cal_auc
from .f1 import cal_f1, cal_image_level_acc, cal_permute_f1


def cal_image_level_auc(logit_list, label_list):
    """
    logit_list: list of probs.max() for each image
    label_list: list of gt.max() for each image
    """
    y_score = np.array(logit_list)
    y_true = np.array([int(gt > 0) for gt in label_list])
    auc = roc_auc_score(y_true, y_score)
    return auc * 100  # Return as a percentage


def cal_image_level_eer(logit_list, label_list):
    """
    logit_list: list of probs.max() for each image
    label_list: list of gt.max() for each image
    """
    y_score = np.array(logit_list)
    y_true = np.array([int(gt > 0) for gt in label_list])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_threshold_index = np.nanargmin(np.absolute(fnr - fpr))
    eer_fpr = fpr[eer_threshold_index]
    eer_fnr = fnr[eer_threshold_index]
    eer = (eer_fpr + eer_fnr) / 2 * 100  # Return as a percentage
    return eer


def crop_inference(image, crop_size, inference_func):
    b, c, ori_height, ori_width = image.shape
    print(image.shape)
    pad_height = max(crop_size[0], ori_height)
    pad_width = max(crop_size[1], ori_width)
    pred = torch.zeros((b, 1, pad_height, pad_width), dtype=torch.float32).cuda()

    height_point = [i * crop_size[0] for i in range(int(pad_height // crop_size[0]))]
    if pad_height % crop_size[0] != 0:
        height_point.append(pad_height - crop_size[0])
    width_point = [i * crop_size[1] for i in range(int(pad_width // crop_size[1]))]
    if pad_width % crop_size[1] != 0:
        width_point.append(pad_width - crop_size[1])

    for y in height_point:
        for x in width_point:
            crop_image = image[..., y : y + crop_size[0], x : x + crop_size[1]]
            """ define process """
            crop_image = F.interpolate(
                crop_image, size=(512, 512), mode="bilinear", align_corners=False
            )
            crop_pred = inference_func(crop_image)
            crop_pred = F.interpolate(
                crop_pred, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            """                """
            pred[..., y : y + crop_size[0], x : x + crop_size[1]] = crop_pred
    return pred[..., :ori_height, :ori_width]


def display(func):
    def calculate_rows_columns(num_images):
        sqrt_num_images = math.sqrt(num_images)
        rows = int(sqrt_num_images)
        columns = int(sqrt_num_images)
        while rows * columns < num_images:
            columns += 1
        return rows, columns

    def wrapper(*args):
        display_name, result = func(*args)
        rows, columns = calculate_rows_columns(len(display_name))
        plt.figure(figsize=(10, 5))
        for i in range(len(display_name)):
            image = result[i]
            plt.subplot(rows, columns, i + 1)
            plt.imshow(
                image, cmap=None if len(result[i].shape) == 3 else "gray"
            ), plt.axis("off"), plt.title(f"{display_name[i]}")
        plt.tight_layout()
        plt.show()
        return result

    return wrapper


@display
def show_result(display_name, result):
    return display_name, result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # Get the batch size N
        N = targets.shape[0]
        # Smoothing variable
        smooth = Variable(torch.ones(1))
        smooth = smooth.cuda()
        # Reshape width and height into a single dimension
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # Compute the intersection
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        # Compute the average loss per image in the batch
        loss = 1 - N_dice_eff.sum() / N
        return loss
