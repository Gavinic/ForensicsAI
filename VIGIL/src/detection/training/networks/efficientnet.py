"""
The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
"""

import argparse
import logging
import math
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from metrics.registry import BACKBONE
from timm import create_model
from torch.nn import init


def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


@BACKBONE.register_module(module_name="efficientnet")
class Efficientnet(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, model_config):
        """Constructor
        Args:
            model_config: configuration file with the dict format
        """
        super(Efficientnet, self).__init__()
        self.num_classes = model_config["num_classes"]
        self.mode = model_config["mode"]
        inc = model_config["inc"]
        self.backbone = create_model(
            self.mode,
            pretrained=model_config["pretrained"],
            num_classes=model_config["num_classes"],
            in_chans=inc,
        )

    def features(self, input):
        x = self.backbone.forward_features(input)
        return x

    def classifier(self, features, id_feat=None):
        out = self.backbone.forward_head(features)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x
