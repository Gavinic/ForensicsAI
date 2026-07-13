import datetime
import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from networks import BACKBONE
from sklearn import metrics
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="replk")
class ReplkDetector(nn.Module):
    def __init__(self, config):
        super(ReplkDetector, self).__init__()
        self.config = config
        self.srm = SRMConv2d_simple().to(dtype=torch.bfloat16)
        self.adapter = nn.Conv2d(
            in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        self.backbone = self.build_backbone(config)
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        if config["pretrained"]:
            state_dict = torch.load(config["pretrained"])
            self.load_state_dict(state_dict)
            print("Load pretrained model successfully!")
            logger.info("Load pretrained model successfully!")

    def build_backbone(self, config):
        backbone_class = BACKBONE[config["backbone_name"]]
        model_config = config["backbone_config"]
        backbone = backbone_class(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 512, 1024],
            drop_path_rate=0.5,
            small_kernel=5,
            num_classes=2,
            use_checkpoint=False,
            small_kernel_merged=False,
            use_sync_bn=True,
        )
        if config["ckpt"]:
            new_weight = backbone.state_dict()
            pre_weight = torch.load(config["ckpt"])
            for k, v in new_weight.items():
                if k in pre_weight and pre_weight[k].shape == v.shape:
                    new_weight[k] = pre_weight[k]
                else:
                    print("skip ", k)
            backbone.load_state_dict(new_weight)
            print("load ckpt ", config["ckpt"])
        backbone = module_init(backbone)
        return backbone

    def features(self, data_dict: dict) -> torch.tensor:
        srm = self.srm(data_dict["image"])
        feat = torch.cat([data_dict["image"], srm], dim=1)
        feat = self.adapter(feat)
        feat = self.backbone(feat)
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return features

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]  # Tensor of shape [batch_size]
        pred = pred_dict["cls"]  # Tensor of shape [batch_size, num_classes]
        loss = self.loss_func(pred, label)
        mask_real = label == 0
        mask_fake = label == 1
        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)
        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)

        loss_dict = {
            "overall": loss,
            "real_loss": loss_real,
            "fake_loss": loss_fake,
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {"cls": pred, "prob": prob, "feat": features}

        return pred_dict


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = torch.from_numpy(self._build_kernel(inc))
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
        # filter2：KV
        filter2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
        # filter3：hor 2rd
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        filter1 = torch.asarray(filter1) / 4.0
        filter2 = np.asarray(filter2) / 12.0
        filter3 = np.asarray(filter3) / 2.0
        # statck the filters
        filters = [
            [filter1],  # , filter1, filter1],
            [filter2],  # , filter2, filter2],
            [filter3],
        ]  # , filter3, filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        return filters


def module_init(model):
    # model = replace_attention_in_vit(model)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "ppw1" in name or "ppw2" in name or "head" in name:
            param.requires_grad = True
    # for name, param in model.named_parameters():
    #     print('{}: {}'.format(name, param.requires_grad))
    return model
