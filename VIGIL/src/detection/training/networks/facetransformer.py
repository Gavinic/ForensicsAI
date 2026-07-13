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
from timm.models.vision_transformer import Attention
from torch.nn import init


@BACKBONE.register_module(module_name="facetransformer")
class FaceTransformer(nn.Module):
    """
    FaceTransformer model
    """

    def __init__(self, config):
        """Constructor
        Args:
            model_config: configuration file with the dict format
        """
        super(FaceTransformer, self).__init__()
        model_config = config["backbone_config"]
        self.num_classes = model_config["num_classes"]
        self.mode = model_config["mode"]
        inc = model_config["inc"]
        self.backbone = create_model(
            self.mode,
            pretrained=True if not config["pretrained"] else False,
            num_classes=model_config["num_classes"],
            in_chans=inc,
        )
        self.backbone = module_init(self.backbone)
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True

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


def module_init(model):
    # model = replace_attention_in_vit(model)
    for param in model.parameters():
        param.requires_grad = False
    for _, module in model.named_children():
        if isinstance(module, Attention):
            module.split_qkv_weights()
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.modules.linear.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(
                        parent_module, sub_module_names[-1], split_linear(sub_module)
                    )
        else:
            module_init(module)
    return model


def split_linear(module):
    if isinstance(module, nn.modules.linear.Linear):
        in_features = module.in_features
        out_features = module.out_features
        # print(out_features)
        bias = module.bias is not None
        new_module = FACELinear(
            in_features, out_features, bias=bias, init_weight=module.weight.data.clone()
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module
    else:
        return module


class FACELinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_weight=None):
        super(FACELinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r1 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.r2 = nn.Parameter(torch.Tensor(1024, 1), requires_grad=True)
        self.r3 = nn.Parameter(torch.Tensor(1, 1024), requires_grad=True)

        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )

        if init_weight is not None:
            U, S, Vh = torch.linalg.svd(init_weight, full_matrices=False)
            U_r = U[:, :1023]
            S_r = S[:1023]
            Vh_r = Vh[:1023, :]
            weight_main = U_r @ torch.diag(S_r) @ Vh_r
            self.weight_main.data.copy_(weight_main)
            self.r1.data.copy_(S[1023:])
            self.r2.data.copy_(U[:, 1023:])
            self.r3.data.copy_(Vh[1023:, :])
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_features), requires_grad=False
            )  # this step needs review
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        residual_weight = self.r2 @ torch.diag(self.r1) @ self.r3
        weight = self.weight_main + residual_weight
        return F.linear(x, weight, self.bias)
