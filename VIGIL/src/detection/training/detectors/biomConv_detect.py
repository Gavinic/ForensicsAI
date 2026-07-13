import datetime
import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, Type, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_mae
from metrics.utils import get_test_metrics, weighted_r2_score
from networks import BACKBONE
from sklearn import metrics
from timm.layers import Mlp
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


def head(combined, hidden, dropout=0.2):
    return nn.Sequential(
        nn.Linear(combined, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


@DETECTOR.register_module(module_name="biomconv")
class BiomConvDetector(nn.Module):
    def __init__(self, config):
        super(BiomConvDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = nn.SmoothL1Loss()  # nn.MSELoss()
        self.loss_mse = nn.MSELoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        feat_dim = self.backbone.num_features
        print("backbone feature dim: ", feat_dim, "split_nums: ", config["split_nums"])
        self.head_total = head(feat_dim, 256)
        self.head_gdm = head(feat_dim, 256)
        self.head_green = head(feat_dim, 256)

        self.head_height = head(feat_dim, 256)
        self.head_hdvi = head(feat_dim, 256)
        self.softplus = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if config["pretrained"]:
            state_dict = torch.load(config["pretrained"])
            self.load_state_dict(state_dict)
            print("Load pretrained model successfully!")
            logger.info("Load pretrained model successfully!")

    def build_backbone(self, config):
        backbone_config = config["backbone_config"]
        backbone = timm.create_model(
            backbone_config["mode"],
            pretrained=backbone_config["pretrained"],
            num_classes=backbone_config["num_classes"],
            # img_size=768
            # in_chans=backbone_config['inc'],
        )
        ## Replace first
        backbone = module_init(backbone)
        for name, param in backbone.named_parameters():
            print("{}: {}".format(name, param.requires_grad))
        return backbone

    def open_allparameter(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True

    def close_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def features(self, data_dict: dict) -> torch.tensor:
        image_left = data_dict["image_left"]  # B, 3, H, W
        image_right = data_dict["image_right"]  # B, 3, H, W
        B, C, H, W = image_left.shape
        h_patch, w_patch = H // 4, W // 4

        # Use unfold to split into blocks: B, 3, H, W -> B, 3*h'*w', 16
        patches_left = F.unfold(
            image_left, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch)
        )
        patches_right = F.unfold(
            image_right, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch)
        )

        # Rearrange: B, 3*h'*w', 16 -> B, 16, 3, h', w'
        patches_left = patches_left.transpose(1, 2).reshape(B, 16, C, h_patch, w_patch)
        patches_right = patches_right.transpose(1, 2).reshape(
            B, 16, C, h_patch, w_patch
        )

        # Merge the batch dimension: B, 16, 3, h', w' -> B*16, 3, h', w'
        patches_left = patches_left.reshape(B * 16, C, h_patch, w_patch)
        patches_right = patches_right.reshape(B * 16, C, h_patch, w_patch)

        f_l = self.backbone.forward_features(patches_left).mean((2, 3))
        f_r = self.backbone.forward_features(patches_right).mean((2, 3))
        f_l = f_l.reshape(B, 16, -1).sum(1)  # restore to original
        f_r = f_r.reshape(B, 16, -1).sum(1)
        g_f_l = self.backbone.forward_features(image_left).mean((2, 3))
        g_f_r = self.backbone.forward_features(image_right).mean((2, 3))
        return f_l, f_r, g_f_l, g_f_r

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss_dict = {}
        overall = 0
        ## Only need to compute part of the loss
        for key, wi in zip(self.config["loss_columns"], self.config["loss_weights"]):
            label = data_dict[key]  # Tensor of shape [batch_size]
            pred = pred_dict[key]  # Tensor of shape [batch_size]
            # print(key, pred,label)
            loss = self.loss_func(pred, label)
            loss_dict[key] = loss
            overall += loss * wi
        ## Similarity between the left and right images
        # sim_loss = (1-pred_dict['lr_sim']).mean()
        # loss_dict['sim_loss'] = sim_loss
        # ## Add two more losses

        hiv_mask = data_dict["Height_Ave_cm"] != 0
        height_loss = self.loss_mse(
            pred_dict["Height_Ave_cm"] * hiv_mask, data_dict["Height_Ave_cm"]
        )
        hdvi_loss = self.loss_mse(
            pred_dict["Pre_GSHH_NDVI"] * hiv_mask, data_dict["Pre_GSHH_NDVI"]
        )
        loss_dict["Height_Ave_cm"] = height_loss
        loss_dict["Pre_GSHH_NDVI"] = hdvi_loss
        overall = overall + hdvi_loss * 2 + height_loss * 2  # + sim_loss*2

        loss_dict["overall"] = overall
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        # compute metrics for batch data
        metric_batch_dict = weighted_r2_score(pred_dict, data_dict)

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        f_l, f_r, g_f_l, g_f_r = self.features(data_dict)
        features = (f_l + g_f_l) / 2 + (f_r + g_f_r) / 2
        # get the prediction by classifier
        green_pos = self.softplus(self.head_green(features))
        total_pos = self.softplus(self.head_total(features))
        gdm_pos = self.softplus(self.head_gdm(features))
        clover_pos = gdm_pos - green_pos
        dead_pos = total_pos - gdm_pos

        height = (
            self.sigmoid(self.head_height(features))
            if "extend" not in data_dict
            else None
        )
        hdvi = (
            self.sigmoid(self.head_hdvi(features))
            if "extend" not in data_dict
            else None
        )
        # build the prediction dict for each output
        # lr_sim = (F.normalize(f_l) * F.normalize(f_r)).sum(dim=1)
        pred_dict = {
            "Dry_Green_g": green_pos,
            "Dry_Clover_g": clover_pos,
            "Dry_Dead_g": dead_pos,
            "Dry_Total_g": total_pos,
            "GDM_g": gdm_pos,
            "Height_Ave_cm": height,
            "Pre_GSHH_NDVI": hdvi,
            #    "lr_sim":lr_sim
        }
        return pred_dict


def module_init(model):
    # model = replace_attention_in_vit(model)
    for param in model.parameters():
        param.requires_grad = False
    for _, module in model.named_children():
        if isinstance(module, Mlp):
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.modules.linear.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    # if 'fc2' in sub_name:
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
        # print(in_features, out_features)
        bias = module.bias is not None
        new_module = ConvLinear(
            in_features, out_features, bias=bias, init_weight=module.weight.data.clone()
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module
    else:
        return module


class ConvLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_weight=None):
        super(ConvLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        r = 1
        # print("****",in_features,out_features )
        self.r1 = nn.Parameter(torch.Tensor(r), requires_grad=True)
        self.r2 = nn.Parameter(torch.Tensor(out_features, r), requires_grad=True)
        self.r3 = nn.Parameter(torch.Tensor(r, in_features), requires_grad=True)

        self.weight_main = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )

        if init_weight is not None:
            U, S, Vh = torch.linalg.svd(init_weight, full_matrices=False)
            # Determine the actual rank
            min_dim = min(out_features, in_features)
            actual_r = min(r, min_dim)
            U_r = U[:, : min_dim - actual_r]
            S_r = S[: min_dim - actual_r]
            Vh_r = Vh[: min_dim - actual_r, :]
            weight_main = U_r @ torch.diag(S_r) @ Vh_r
            self.weight_main.data.copy_(weight_main)

            # Residual components (the last r)
            self.r1.data.copy_(S[min_dim - actual_r : min_dim - actual_r + actual_r])
            self.r2.data.copy_(U[:, min_dim - actual_r : min_dim - actual_r + actual_r])
            self.r3.data.copy_(
                Vh[min_dim - actual_r : min_dim - actual_r + actual_r, :]
            )
            # self.r1.data.copy_(S[min_dim-actual_r:] )
            # self.r2.data.copy_(U[:, min_dim-actual_r:])
            # self.r3.data.copy_(Vh[min_dim-actual_r:, :])
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
