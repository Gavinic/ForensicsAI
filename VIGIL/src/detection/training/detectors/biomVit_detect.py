import datetime
import logging
import math
import os
import random
from collections import defaultdict

# from timm.models.vision_transformer_sam import Attention, get_decomposed_rel_pos_bias,apply_rot_embed_cat
from types import MethodType
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
from timm.models.eva import EvaAttention
from timm.models.vision_transformer import Attention
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


def head(combined, hidden, output=1, dropout=0.2):
    return nn.Sequential(
        nn.Linear(combined, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, output),
    )


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(
                dim
            ),  # LayerNorm is usually more stable than BatchNorm for regression tasks
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)  # residual connection


def advanced_head(input_dim, hidden_dim, output_dim=1, num_blocks=2, dropout=0.1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    # Stack residual blocks
    for _ in range(num_blocks):
        layers.append(ResBlock(hidden_dim, dropout))

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class StereoFusionNeck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Assume the input feature is [B, dim]
        # We have 4 sources: s_f_l, g_f_l, s_f_r, g_f_r
        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        # Optional: introduce Cross-Attention to let the left and right images interact
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, s_f_l, s_f_r, g_f_l, g_f_r):
        # Simple concatenation fusion
        cat_feat = torch.cat([s_f_l, s_f_r, g_f_l, g_f_r], dim=1)  # [B, dim*4]
        fused = self.fusion(cat_feat)
        return fused


class BiomassHead(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 1. Predict total dry mass (Total Dry Mass)
        self.total_head = advanced_head(in_dim, hidden_dim, output_dim=1, dropout=0.2)
        # 1. Predict green dry mass (Green Dry Mass)
        self.green_head = advanced_head(in_dim, hidden_dim, output_dim=1, dropout=0.2)
        # Predict GDM
        self.gdm_head = advanced_head(in_dim, hidden_dim, output_dim=1, dropout=0.2)

        # 2. Predict component ratios (Green vs Dead) -> 2 classes
        self.dead_head = nn.Sequential(
            advanced_head(in_dim, hidden_dim, output_dim=1, dropout=0.2), nn.Sigmoid()
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Predict the total mass (ensure it is positive)
        total_mass = self.softplus(self.total_head(x))
        green_mass = self.softplus(self.green_head(x))
        gdm_mass = self.softplus(self.gdm_head(x))
        # Predict the live/dead ratio [p_green, p_dead]
        ratios_dead_live = self.dead_head(x)
        dead_mass = total_mass * ratios_dead_live
        clover_mass = gdm_mass - green_mass

        # Predict the specific component ratios in the live mass (e.g. GDM, Clover)
        return {
            "Dry_Total_g": total_mass,
            "Dry_Green_g": green_mass,
            "Dry_Dead_g": dead_mass,
            "GDM_g": gdm_mass,
            "Dry_Clover_g": clover_mass,
        }


class MultiTaskBiomassHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()

        # 1. Shared trunk - extract generic biological features
        # The in_dim here should be the concatenated dimension (e.g. feat_dim * 4)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm is recommended for regression tasks
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Task-specific branches
        # A simple Linear is enough, since the trunk already performs non-linear transformation
        self.head_total = nn.Linear(hidden_dim, 1)  # total mass
        self.head_green = nn.Linear(hidden_dim, 1)  # green-grass ratio (use Sigmoid)
        self.head_gdm = nn.Linear(hidden_dim, 1)  # GDM ratio within green grass
        self.head_dead = nn.Linear(hidden_dim, 1)  # GDM ratio within green grass

        self.head_height = nn.Linear(hidden_dim, 1)
        self.head_ndvi = nn.Linear(hidden_dim, 1)

        # Classification tasks
        self.head_state = nn.Linear(hidden_dim, 4)
        self.head_species = nn.Linear(hidden_dim, 16)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()  # ensure the output > 0

    def forward(self, x):
        # x shape: [Batch, feat_dim * 4]
        feat = self.trunk(x)

        # --- Physics-informed constraints ---

        # 1. Predict total dry mass (Total Dry Mass) -> must be greater than 0
        total_mass = self.softplus(self.head_total(feat))
        green_mass = self.softplus(self.head_green(feat))
        gdm_mass = self.softplus(self.head_gdm(feat))
        ratios_dead_live = self.sigmoid(self.head_dead(feat))
        dead_mass = total_mass * ratios_dead_live
        clover_mass = gdm_mass - green_mass

        # Other metrics
        height = self.sigmoid(
            self.head_height(feat)
        )  # activation can be omitted here, or use ReLU
        ndvi = self.sigmoid(self.head_ndvi(feat))  # NDVI is also 0-1 (usually)

        state = self.head_state(feat)
        species = self.head_species(feat)

        return {
            "Dry_Total_g": total_mass,
            "Dry_Green_g": green_mass,
            "Dry_Dead_g": dead_mass,
            "Dry_Clover_g": clover_mass,
            "GDM_g": gdm_mass,
            "Height_Ave_cm": height,
            "Pre_GSHH_NDVI": ndvi,
            "state": state,
            "species": species,
        }


@DETECTOR.register_module(module_name="biomvit")
class BiomVitDetector(nn.Module):
    def __init__(self, config):
        super(BiomVitDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = nn.SmoothL1Loss()  # nn.MSELoss()
        self.loss_func_non = nn.SmoothL1Loss(reduction="None")
        self.loss_mse = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        feat_dim = self.backbone.num_features
        print("backbone feature dim: ", feat_dim, "split_nums: ", config["split_nums"])
        # self.head_total = head(feat_dim, 256)
        # self.head_gdm = head(feat_dim,256)
        # self.head_green = head(feat_dim, 256)
        # self.head_dead_rate = head(feat_dim, 256)
        self.biom_head = MultiTaskBiomassHead(feat_dim, 128)
        # BiomassHead(feat_dim, 128)

        self.head_height = head(feat_dim, 256)
        self.head_hdvi = head(feat_dim, 256)
        self.head_sate = head(feat_dim, 256, 4)
        self.head_spes = head(feat_dim, 256, 16)
        # self.adapter = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.softplus = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def build_backbone(self, config):
        backbone_config = config["backbone_config"]
        backbone = timm.create_model(
            backbone_config["mode"],
            pretrained=backbone_config["pretrained"],
            num_classes=backbone_config["num_classes"],
            # img_size=768,
            # in_chans=6#backbone_config['inc'],
        )
        backbone = module_init(backbone)
        if config["pretrained"]:
            state_dict = torch.load(config["pretrained"])
            backbone.load_state_dict(state_dict)
            print("Load pretrained model successfully!")
            logger.info("Load pretrained model successfully!")

        for name, param in backbone.named_parameters():
            print("{}: {}".format(name, param.requires_grad))
        ## Replace first
        return backbone

    def open_allparameter(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True
            # print('{}: {}'.format(name, param.requires_grad))

    def features(self, data_dict: dict) -> torch.tensor:
        image_left = data_dict["image_left"]  # B, 3, H, W
        image_right = data_dict["image_right"]  # B, 3, H, W
        # mask_left = data_dict['mask_left']
        # mask_right = data_dict['mask_right'] # B,h,W,2
        # image_left = self.adapter(torch.cat([image_left, mask_left.permute(0,3,1,2)], dim=1))
        # image_right = self.adapter(torch.cat([image_right, mask_right.permute(0,3,1,2)], dim=1))
        B, C, H, W = image_left.shape
        # ========== Small patches (256x256): process in 4 batches ==========
        h_patch, w_patch = H // 4, W // 4
        patches_left = F.unfold(
            image_left, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch)
        )
        patches_right = F.unfold(
            image_right, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch)
        )

        patches_left = patches_left.transpose(1, 2).reshape(B, 16, C, h_patch, w_patch)
        patches_right = patches_right.transpose(1, 2).reshape(
            B, 16, C, h_patch, w_patch
        )
        patches_left = patches_left.reshape(B * 16, C, h_patch, w_patch)
        patches_right = patches_right.reshape(B * 16, C, h_patch, w_patch)
        # Use checkpoint to save GPU memory
        s_f_l = checkpoint(self.backbone, patches_left, use_reentrant=False)
        s_f_r = checkpoint(self.backbone, patches_right, use_reentrant=False)
        # s_f_l = self.backbone(patches_left)
        # s_f_r = self.backbone(patches_right)
        s_f_l = s_f_l.reshape(B, 16, -1)
        s_f_r = s_f_r.reshape(B, 16, -1)
        # ========== Global features ==========
        # Use checkpoint to save GPU memory
        g_f_l = checkpoint(self.backbone, image_left, use_reentrant=False)
        g_f_r = checkpoint(self.backbone, image_right, use_reentrant=False)
        # g_f_l = self.backbone(image_left)
        # g_f_r = self.backbone(image_right)

        # ========== Medium patches (512x512): process in batches ==========
        # h_patch, w_patch = H // 2, W // 2
        # patches_left = F.unfold(image_left, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch))
        # patches_right = F.unfold(image_right, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch))

        # patches_left = patches_left.transpose(1, 2).reshape(B, 4, C, h_patch, w_patch)
        # patches_right = patches_right.transpose(1, 2).reshape(B, 4, C, h_patch, w_patch)
        # patches_left = patches_left.reshape(B * 4, C, h_patch, w_patch)
        # patches_right = patches_right.reshape(B * 4, C, h_patch, w_patch)
        # b_f_l = checkpoint(self.backbone, patches_left, use_reentrant=False)
        # b_f_r = checkpoint(self.backbone, patches_right, use_reentrant=False)
        # b_f_l = b_f_l.reshape(B, 4, -1)
        # b_f_r = b_f_r.reshape(B, 4, -1)

        return s_f_l.sum(1), s_f_r.sum(1), g_f_l, g_f_r

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
        ## Add two more losses

        hiv_mask = data_dict["Height_Ave_cm"] != -1
        height_loss = self.loss_mse(
            pred_dict["Height_Ave_cm"] * hiv_mask, data_dict["Height_Ave_cm"]
        )
        hdvi_loss = self.loss_mse(
            pred_dict["Pre_GSHH_NDVI"] * hiv_mask, data_dict["Pre_GSHH_NDVI"]
        )
        loss_dict["Height_Ave_cm"] = height_loss
        loss_dict["Pre_GSHH_NDVI"] = hdvi_loss
        overall = overall + hdvi_loss * 10 + height_loss * 10  # + sim_loss*2

        ## Add the classification loss
        mask_state = data_dict["State"] != -1
        # print(mask_state.shape,pred_dict['state'].shape)
        mask_species = data_dict["Species"] != -1
        # print(mask_species.shape,pred_dict['species'].shape)
        sate_loss = self.loss_ce(
            pred_dict["state"] * mask_state.unsqueeze(-1),
            data_dict["State"] * mask_state,
        )
        spes_loss = self.loss_bce(
            pred_dict["species"] * mask_species, data_dict["Species"] * mask_species
        )
        loss_dict["state"] = sate_loss
        loss_dict["species"] = spes_loss
        # loss_dict['f_mse_loss'] = pred_dict['f_mse_loss']
        overall = overall + sate_loss * 1 + spes_loss * 2
        loss_dict["overall"] = overall

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        # compute metrics for batch data
        metric_batch_dict = weighted_r2_score(pred_dict, data_dict)

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        s_f_l, s_f_r, g_f_l, g_f_r = self.features(data_dict)

        features = (s_f_l + g_f_l + s_f_r + g_f_r) / 2

        # lr_sim = (F.normalize(g_f_l) * F.normalize(g_f_r)).sum(dim=1)
        ## Features must be consistent!
        # f_mse = self.loss_mse(s_f_l, g_f_l) + self.loss_mse(s_f_r, g_f_r)
        # get the prediction by classifier
        # green_pos = self.softplus(self.head_green(features))
        # total_pos = self.softplus(self.head_total(features))
        # gdm_pos = self.softplus(self.head_gdm(features))

        # dead_pos = self.sigmoid(self.head_dead_rate(features)) * total_pos
        # clover_pos = gdm_pos - green_pos
        # dead_pos = total_pos - gdm_pos
        pred_dict = self.biom_head(features)
        # pred_dict['state'] = self.head_sate(features)
        # pred_dict['species'] = self.head_spes(features)
        # pred_dict['Height_Ave_cm'] = self.sigmoid(self.head_height(g_f_l+g_f_r)) if 'extend' not in data_dict else None
        # pred_dict['Pre_GSHH_NDVI'] = self.sigmoid(self.head_hdvi(g_f_l+g_f_r)) if 'extend' not in data_dict else None

        # build the prediction dict for each output

        # pred_dict = {"Dry_Green_g":green_pos,"Dry_Clover_g":clover_pos,
        #              "Dry_Dead_g":dead_pos,"Dry_Total_g":total_pos, "GDM_g":gdm_pos,
        #              "Height_Ave_cm":height,"Pre_GSHH_NDVI":hdvi,
        #              'state':state,'species':spes #, "f_mse_loss":f_mse
        #    "lr_sim":lr_sim
        #  }

        return pred_dict


def module_init(model):
    # model = replace_attention_in_vit(model)
    for param in model.parameters():
        param.requires_grad = False
    for _, module in model.named_children():
        if isinstance(module, EvaAttention):
            is_qkv_fused = module.qkv is not None
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.modules.linear.Linear) and "q" in sub_name:
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        split_linear(sub_module, is_qkv_fused),
                    )
        else:
            module_init(module)
    return model


def split_linear(module, is_qkv_fused):
    if isinstance(module, nn.modules.linear.Linear):
        in_features = module.in_features
        out_features = module.out_features
        # print(out_features)
        bias = module.bias is not None
        new_module = FACELinear(
            in_features,
            out_features,
            bias=bias,
            is_qkv_fused=is_qkv_fused,
            init_weight=module.weight.data.clone(),
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
        return new_module
    else:
        return module


class FACELinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, init_weight=None, is_qkv_fused=False
    ):
        super(FACELinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_qkv_fused = is_qkv_fused
        r = 1

        if is_qkv_fused and out_features % 3 == 0:
            # QKV-fused case: handle the q, k, v parts separately
            attn_dim = out_features // 3

            # Create low-rank parameters for q, k, v separately
            self.r1_q = nn.Parameter(torch.Tensor(r), requires_grad=True)
            self.r2_q = nn.Parameter(torch.Tensor(attn_dim, r), requires_grad=True)
            self.r3_q = nn.Parameter(torch.Tensor(r, in_features), requires_grad=True)

            self.r1_k = nn.Parameter(torch.Tensor(r), requires_grad=True)
            self.r2_k = nn.Parameter(torch.Tensor(attn_dim, r), requires_grad=True)
            self.r3_k = nn.Parameter(torch.Tensor(r, in_features), requires_grad=True)

            self.r1_v = nn.Parameter(torch.Tensor(r), requires_grad=True)
            self.r2_v = nn.Parameter(torch.Tensor(attn_dim, r), requires_grad=True)
            self.r3_v = nn.Parameter(torch.Tensor(r, in_features), requires_grad=True)

            # Main weight matrix (frozen)
            self.weight_main = nn.Parameter(
                torch.Tensor(out_features, in_features), requires_grad=False
            )

            if init_weight is not None:
                # Split the qkv weights
                q_weight = init_weight[:attn_dim, :]
                k_weight = init_weight[attn_dim : 2 * attn_dim, :]
                v_weight = init_weight[2 * attn_dim :, :]

                # Perform SVD decomposition on q, k, v separately
                min_dim = min(attn_dim, in_features)
                actual_r = min(r, min_dim - 1)

                # SVD of Q
                U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
                if min_dim > actual_r:
                    q_main = (
                        U_q[:, : min_dim - actual_r]
                        @ torch.diag(S_q[: min_dim - actual_r])
                        @ Vh_q[: min_dim - actual_r, :]
                    )
                    self.r1_q.data.copy_(
                        S_q[min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r2_q.data.copy_(
                        U_q[:, min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r3_q.data.copy_(
                        Vh_q[min_dim - actual_r : min_dim - actual_r + actual_r, :]
                    )
                else:
                    q_main = q_weight
                    nn.init.normal_(self.r1_q, std=0.01)
                    nn.init.normal_(self.r2_q, std=0.01)
                    nn.init.normal_(self.r3_q, std=0.01)

                # SVD of K
                U_k, S_k, Vh_k = torch.linalg.svd(k_weight, full_matrices=False)
                if min_dim > actual_r:
                    k_main = (
                        U_k[:, : min_dim - actual_r]
                        @ torch.diag(S_k[: min_dim - actual_r])
                        @ Vh_k[: min_dim - actual_r, :]
                    )
                    self.r1_k.data.copy_(
                        S_k[min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r2_k.data.copy_(
                        U_k[:, min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r3_k.data.copy_(
                        Vh_k[min_dim - actual_r : min_dim - actual_r + actual_r, :]
                    )
                else:
                    k_main = k_weight
                    nn.init.normal_(self.r1_k, std=0.01)
                    nn.init.normal_(self.r2_k, std=0.01)
                    nn.init.normal_(self.r3_k, std=0.01)

                # SVD of V
                U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)
                if min_dim > actual_r:
                    v_main = (
                        U_v[:, : min_dim - actual_r]
                        @ torch.diag(S_v[: min_dim - actual_r])
                        @ Vh_v[: min_dim - actual_r, :]
                    )
                    self.r1_v.data.copy_(
                        S_v[min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r2_v.data.copy_(
                        U_v[:, min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r3_v.data.copy_(
                        Vh_v[min_dim - actual_r : min_dim - actual_r + actual_r, :]
                    )
                else:
                    v_main = v_weight
                    nn.init.normal_(self.r1_v, std=0.01)
                    nn.init.normal_(self.r2_v, std=0.01)
                    nn.init.normal_(self.r3_v, std=0.01)

                # Combine into the full main weight matrix
                self.weight_main.data = torch.cat([q_main, k_main, v_main], dim=0)
            else:
                nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
                nn.init.normal_(self.r1_q, std=0.01)
                nn.init.normal_(self.r2_q, std=0.01)
                nn.init.normal_(self.r3_q, std=0.01)
                nn.init.normal_(self.r1_k, std=0.01)
                nn.init.normal_(self.r2_k, std=0.01)
                nn.init.normal_(self.r3_k, std=0.01)
                nn.init.normal_(self.r1_v, std=0.01)
                nn.init.normal_(self.r2_v, std=0.01)
                nn.init.normal_(self.r3_v, std=0.01)
        else:
            # Ordinary linear-layer case
            self.r1 = nn.Parameter(torch.Tensor(r), requires_grad=True)
            self.r2 = nn.Parameter(torch.Tensor(out_features, r), requires_grad=True)
            self.r3 = nn.Parameter(torch.Tensor(r, in_features), requires_grad=True)

            self.weight_main = nn.Parameter(
                torch.Tensor(out_features, in_features), requires_grad=False
            )

            if init_weight is not None:
                U, S, Vh = torch.linalg.svd(init_weight, full_matrices=False)
                min_dim = min(out_features, in_features)
                actual_r = min(r, min_dim - 1)

                if min_dim > actual_r:
                    U_r = U[:, : min_dim - actual_r]
                    S_r = S[: min_dim - actual_r]
                    Vh_r = Vh[: min_dim - actual_r, :]
                    weight_main = U_r @ torch.diag(S_r) @ Vh_r
                    self.weight_main.data.copy_(weight_main)
                    self.r1.data.copy_(
                        S[min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r2.data.copy_(
                        U[:, min_dim - actual_r : min_dim - actual_r + actual_r]
                    )
                    self.r3.data.copy_(
                        Vh[min_dim - actual_r : min_dim - actual_r + actual_r, :]
                    )
                else:
                    self.weight_main.data.copy_(init_weight)
                    nn.init.normal_(self.r1, std=0.01)
                    nn.init.normal_(self.r2, std=0.01)
                    nn.init.normal_(self.r3, std=0.01)
            else:
                nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
                nn.init.normal_(self.r1, std=0.01)
                nn.init.normal_(self.r2, std=0.01)
                nn.init.normal_(self.r3, std=0.01)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.is_qkv_fused:
            # Compute the residual weights of q, k, v separately, then concatenate
            attn_dim = self.out_features // 3

            residual_q = self.r2_q @ torch.diag(self.r1_q) @ self.r3_q
            residual_k = self.r2_k @ torch.diag(self.r1_k) @ self.r3_k
            residual_v = self.r2_v @ torch.diag(self.r1_v) @ self.r3_v

            residual_weight = torch.cat([residual_q, residual_k, residual_v], dim=0)
        else:
            residual_weight = self.r2 @ torch.diag(self.r1) @ self.r3

        weight = self.weight_main + residual_weight
        return F.linear(x, weight, self.bias)


def new_attention_forward(self, x, attn_mask=None):

    B, N, C = x.shape
    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    # q, k, v = qkv.unbind(0)
    q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
