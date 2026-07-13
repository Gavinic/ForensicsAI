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


@DETECTOR.register_module(module_name="orth")
class OrthDetector(nn.Module):
    def __init__(self, config):
        super(OrthDetector, self).__init__()
        self.config = config
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
        backbone = backbone_class(config)
        return backbone

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone.features(data_dict["image"])
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)

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
