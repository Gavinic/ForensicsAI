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
from metrics.utils import pixel_label_f1
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
        nn.GELU(),
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)  # residual connection


def advanced_head(input_dim, hidden_dim, output_dim=1, num_blocks=2, dropout=0.1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    # Stack residual blocks
    for _ in range(num_blocks):
        layers.append(ResBlock(hidden_dim, dropout))

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Basic conv block: Conv + BN + ReLU"""

    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Use bilinear interpolation + convolution instead of deconvolution
    This reduces checkerboard artifacts and makes the mask smoother
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x):
        # 2x upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class ImprovedIdmbHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=128):
        super().__init__()

        # -----------------------------------------------------------
        # 1. Feature dimensionality reduction and context enhancement (for the 1024-dim features of ViT-Large)
        # -----------------------------------------------------------
        self.reduce_conv = ConvBlock(in_dim, 256, kernel_size=1, padding=0)
        # Simple context module that enlarges the receptive field and removes isolated noise
        # If GPU memory allows, this can be replaced with standard ASPP
        self.context_block = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, padding=2, dilation=2, bias=False
            ),  # dilated convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # -----------------------------------------------------------
        # 2. Mask prediction branch (Progressive Upsampling)
        # -----------------------------------------------------------
        # Assume the ViT output is 1/16; we need to upsample 4 times to return to the original size (or close to it)
        # Structure: [256 -> 128] -> [128 -> 64] -> [64 -> 32] -> [32 -> 1]

        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)

        # The last layer has no BN or ReLU and directly outputs logits
        self.final_mask = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # -----------------------------------------------------------
        # 3. Classification prediction branch (Label Head) - enhanced version
        # -----------------------------------------------------------
        # We not only use the CLS token but also combine it with Global Max Pooling of the feature map
        # Since forgery is usually local, Max Pooling can capture the strongest forgery signal
        self.cls_head = advanced_head(in_dim + in_dim, 128, output_dim=1, dropout=0.2)

    def forward(self, feature_map, feature_cls):
        # feature_map: (B, 1024, H/16, W/16)
        # feature_cls: (B, 1024)

        # --- Mask branch ---
        # 1. Dimensionality reduction
        x = self.reduce_conv(feature_map)

        # 2. Context aggregation (key: helps remove non-connected noise)
        x = x + self.context_block(x)  # residual connection

        # 3. Extract features for the classification head (Global Max Pooling)
        # The x here contains spatial forgery features; taking Max can capture 'whether a forgery region exists'
        # shape: (B, 256, H', W') -> (B, 256)
        feat_max = F.adaptive_max_pool2d(feature_map, (1, 1)).view(x.size(0), -1)

        # 4. Progressively upsample to generate the mask
        d1 = self.up1(x)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        pred_mask_logits = self.final_mask(d3)  # (B, 1, H, W)

        # --- Label branch (feature fusion) ---
        # Concatenate the CLS token and the Max Pooling of the feature map
        # This gives the classification head both global semantics (ViT CLS) and the strongest local forgery feature (Conv Max)
        combined_feat = torch.cat([feature_cls, feat_max], dim=1)  # 1024 + 256
        pred_label_logits = self.cls_head(combined_feat).squeeze(1)

        return {"pred_mask": pred_mask_logits, "pred_label": pred_label_logits}


class PositionEmbeddingSine(nn.Module):
    """
    Faithful DETR 2D sine positional encoding
    Mechanism-aligned: replaces SinePositionalEncoding in the config
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        mask = torch.ones((B, H, W), device=x.device, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 256, H, W)
        return pos


# class UpsampleBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.GELU()
#         )
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         return self.conv(x)


class DinoStyleSemanticHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, num_queries=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # 1. Dimensionality-reduction projection
        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)

        # 2. Positional encoder
        self.pe_layer = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True
        )

        # 3. Pixel Decoder - used to generate the base features for the final HD mask
        self.pixel_up1 = UpsampleBlock(hidden_dim, 128)
        self.pixel_up2 = UpsampleBlock(128, 64)
        self.pixel_up3 = UpsampleBlock(64, 32)
        self.pixel_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )  # nn.Conv2d(32, 32, kernel_size=1)

        # 4. Core DINO/DETR mechanism: Query definition
        # query_embed acts as the position/reference-point feature, query_feat as the content feature
        self.query_embed = nn.Linear(
            in_dim, hidden_dim
        )  # nn.Embedding(num_queries, hidden_dim)

        # 5. Transformer Decoder (mechanism-aligned: self-attention -> cross-attention -> FFN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 6. Output projection layers
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),  # output dimension matches pixel_proj
        )

        self.cls_head = advanced_head(
            in_dim + hidden_dim, 128, output_dim=1, dropout=0.2
        )

    def forward(self, feature_map, feature_cls):
        """
        feature_map: (B, 1024, H, W)
        feature_cls: (B, 1024)
        """
        B, _, H, W = feature_map.shape

        # --- Prepare the Transformer Key/Value and PE ---
        src = self.input_proj(feature_map)  # (B, 256, H, W)
        pos_embed = self.pe_layer(src)  # (B, 256, H, W)

        # Flatten spatial dimensions for Transformer
        # TransformerDecoder in batch_first=True expects (B, SeqLen, Dim)
        memory = src.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        memory_pos = pos_embed.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)

        # Add the positional encoding to the memory (standard DETR practice)
        memory = memory + memory_pos

        # --- Prepare the Query ---
        # Mechanism-aligned: Query content is initialized to 0 (tgt); only query_pos guides the search
        # tgt = torch.zeros(B, self.num_queries, self.hidden_dim, device=src.device)
        # query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, N, 256)
        query_pos = self.query_embed(feature_cls)  # B,5,32

        # --- Transformer Decoding (core of the Query mechanism) ---
        # Queries perform Self-Attention among themselves, then Cross-Attention with the global image
        # In the standard PyTorch implementation, query_pos is added to tgt
        out_queries = self.transformer_decoder(
            tgt=query_pos, memory=memory
        )  # (B, N, 256)

        # --- Dynamic mask generation (mechanism-aligned: replaces BBox regression) ---
        p1 = self.pixel_up1(src)
        p2 = self.pixel_up2(p1)
        p3 = self.pixel_up3(p2)
        pixel_features = self.pixel_proj(p3)  # (B, 32, H_out, W_out)

        mask_weights = self.mask_embed(out_queries)  # (B, N, 32)

        # Einstein summation: the weight vectors of N queries dot the HD pixel feature map -> N predicted masks
        pred_masks = torch.einsum("bnc,bchw->bnhw", mask_weights, pixel_features)

        # We perform single-class semantic segmentation; take the strongest response found by all queries
        final_mask = pred_masks.max(dim=1, keepdim=True)[0]  # (B, 1, H_out, W_out)

        # --- Label classification ---
        query_feat_max = out_queries[:, 0]  # .max(dim=1)[0]  # the first query
        combined_feat = torch.cat(
            [feature_cls[:, 0], query_feat_max], dim=1
        )  # the first for classification
        pred_label_logits = self.cls_head(combined_feat).squeeze(1)

        return {"pred_mask": final_mask, "pred_label": pred_label_logits}


class IdmbHead(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 1. Predict whether it is forged
        self.label_head = advanced_head(in_dim, hidden_dim, output_dim=1, dropout=0.2)
        # 2. Predict the forgery mask
        self.mask_head = self.deconv_model = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: 1, 256, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: 1, 128, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: 1, 64, 256, 256
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output: 1, 1, 512, 512
        )

    def forward(self, feature_map, feature_cls):
        pred_label = self.label_head(feature_cls).squeeze(1)
        pred_mask = self.mask_head(feature_map)
        return {"pred_mask": pred_mask, "pred_label": pred_label}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (before sigmoid)
        # targets: binary labels
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # p_t

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # strength of penalizing FP (False Positive); higher suppresses noise more
        self.beta = beta  # strength of penalizing FN (False Negative)
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits
        probs = torch.sigmoid(inputs)

        # Flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(1)
        FP = ((1 - targets) * probs).sum(1)
        FN = (targets * (1 - probs)).sum(1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        # pred is logits (before sigmoid)
        probs = torch.sigmoid(pred)

        # 1. BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # 2. Dice Loss
        smooth = 1e-6
        # Flatten to (Batch, -1)
        p = probs.view(probs.size(0), -1)
        t = target.view(target.size(0), -1)

        intersection = (p * t).sum(-1)
        dice = 1 - (2.0 * intersection + smooth) / (p.sum(-1) + t.sum(-1) + smooth)

        return self.bce_weight * bce + self.dice_weight * dice.mean()


@DETECTOR.register_module(module_name="idbmvit")
class IdbmVitDetector(nn.Module):
    def __init__(self, config):
        super(IdbmVitDetector, self).__init__()
        self.config = config
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_combined = CombinedLoss(dice_weight=0.8, bce_weight=1.0)
        # For the case of very few positive samples, Focal Loss alpha is recommended to be larger (e.g. 0.75), gamma = 2
        self.focal = FocalLoss(alpha=0.2, gamma=2.0)

        # For the case of many false masks, set Tversky alpha (FP weight) larger (e.g. 0.7)
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)

        self.backbone = self.build_backbone(config)
        feat_dim = self.backbone.num_features
        print("backbone feature dim: ", feat_dim)
        self.feat_dim = feat_dim
        self.cotset = {}
        self.backbone.patch_embed.strict_img_size = False

        self.idmb_head = ImprovedIdmbHead(feat_dim, 128)  # IdmbHead(feat_dim, 128)

        self.sigmoid = nn.Sigmoid()
        if config["pretrained"]:
            self.load_state_dict(torch.load(config["pretrained"]))
            print("Pretrained weights loaded successfully")
        if config["all_parameter"]:
            print("Enable full-parameter training")
            self.open_allparameter()  # enable

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
        ##

        # for name, param in backbone.named_parameters():
        #     print('{}: {}'.format(name, param.requires_grad))
        ## Replace first
        return backbone

    def open_allparameter(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True
            # print('{}: {}'.format(name, param.requires_grad))

    def features(self, data_dict: dict) -> torch.tensor:
        image = data_dict["image"]  # B, 3, H, W
        feature = self.backbone.forward_features(image)  # (b, patch+1, dim)
        h = w = int((feature.shape[1] - 1) ** 0.5)
        feature = feature.permute(0, 2, 1)
        feature_map = feature[:, :, 5:].reshape(image.shape[0], self.feat_dim, h, w)
        feature_cls = feature[:, :, 0]  # (b, dim)

        return feature_map, feature_cls

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss_dict = {}
        overall = 0
        label_loss = self.loss_bce(pred_dict["pred_label"], data_dict["label"])
        mask_loss = self.loss_combined(pred_dict["pred_mask"], data_dict["gt_mask"])
        tversky_loss = self.tversky(pred_dict["pred_mask"], data_dict["gt_mask"])
        loss_dict["label_loss"] = label_loss * 2
        loss_dict["mask_loss"] = mask_loss * 0.25
        loss_dict["tversky_loss"] = tversky_loss * 0.4
        overall = label_loss * 2 + (mask_loss + tversky_loss)
        loss_dict["overall"] = overall
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        # compute metrics for batch data
        metric_batch_dict = pixel_label_f1(pred_dict, data_dict)

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        feature_map, feature_cls = self.features(data_dict)
        pred_dict = self.idmb_head(feature_map, feature_cls)

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
