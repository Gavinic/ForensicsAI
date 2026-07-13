import math
import os

import torch
import torch.nn as nn


class ForgeryDetector(nn.Module):
    def __init__(self, input_channels=256, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, C)
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_map):
        """
        feat_map: tensor of shape (B, C, H, W), from SAM encoder
        returns: logits of shape (B, 2)
        """
        x = self.pool(feat_map)
        logits = self.classifier(x)
        return logits


class LoRA_QKV(nn.Module):
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = True
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        if self.activate:
            qkv[:, :, :, : self.dim] += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim : -self.dim] += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim :] += self.b_v(self.a_v(x))
        return qkv


class LoRA_FFN(nn.Module):
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = True
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out


class Adv_LoRA_QKV(nn.Module):
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = False
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        # adversarial expert
        if self.activate:
            qkv[:, :, :, : self.dim] += self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim : -self.dim] += self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim :] += self.b_v(self.a_v(x))

        return qkv


class Adv_LoRA_FFN(nn.Module):
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = False
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out


class ForensicsSAM(nn.Module):
    def __init__(
        self,
        sam_model,
        r=8,
        lora_layer=None,
        forgery_experts_path="./weight/forgery_experts.pth",
        adversary_experts_path="./weight/adversary_experts.pth",
        load_pretrained=True,
        freeze_shared_experts=True,
        freeze_detector=True,
        enable_adversary_experts=True,
    ):
        super().__init__()
        assert r > 0
        self.lora_layer = lora_layer or list(range(len(sam_model.image_encoder.blocks)))

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        blk_num = len(sam_model.image_encoder.blocks)
        if blk_num == 32:
            self.global_attn_index = [7, 15, 23, 31]
        elif blk_num == 24:
            self.global_attn_index = [5, 11, 17, 23]
        elif blk_num == 12:
            self.global_attn_index = [2, 5, 8, 11]

        """ Shared Forgery Experts """
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            blk.attn.qkv = LoRA_QKV(w_qkv_linear, dim, r)

            w_lin1 = blk.mlp.lin1
            in_features = w_lin1.in_features
            out_features = w_lin1.out_features
            blk.mlp.lin1 = LoRA_FFN(w_lin1, in_features, out_features, r)

            if freeze_shared_experts:
                for param in blk.attn.qkv.parameters():
                    param.requires_grad = False
                for param in blk.mlp.lin1.parameters():
                    param.requires_grad = False
            else:
                for param in blk.attn.qkv.ori_qkv.parameters():
                    param.requires_grad = False
                for param in blk.mlp.lin1.ori_lin.parameters():
                    param.requires_grad = False

        # Freeze prompt_encoder and mask_decoder
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        """ Forgery Detector"""
        self.detector = ForgeryDetector()
        if freeze_detector:
            for param in self.detector.parameters():
                param.requires_grad = False

        self.sam = sam_model
        if load_pretrained and forgery_experts_path is not None:
            self.load_all_parameters(forgery_experts_path)

        self.enable_adversary_experts = enable_adversary_experts

        """ Adaptive Adversarial Experts """
        if self.enable_adversary_experts:
            for blk in [
                self.sam.image_encoder.blocks[i] for i in self.global_attn_index
            ]:
                w_qkv_linear = blk.attn.qkv
                blk.attn.qkv = Adv_LoRA_QKV(w_qkv_linear, dim, 8 * r)

                w_lin1 = blk.mlp.lin1
                blk.mlp.lin1 = Adv_LoRA_FFN(w_lin1, in_features, out_features, 8 * r)
            if load_pretrained and adversary_experts_path is not None:
                self.load_all_parameters(adversary_experts_path)

    def activate_adv(self, activate):
        if not self.enable_adversary_experts:
            return
        for blk in [self.sam.image_encoder.blocks[i] for i in self.global_attn_index]:
            blk.attn.qkv.set_activate(activate)
            blk.mlp.lin1.set_activate(activate)

    @staticmethod
    def _to_activate_flag(activate_adv):
        if isinstance(activate_adv, torch.Tensor):
            if activate_adv.numel() == 1:
                return bool(activate_adv.item())
            return bool(torch.any(activate_adv > 0).item())
        return bool(activate_adv)

    def forward(self, images, activate_adv):
        activate_flag = (
            self._to_activate_flag(activate_adv)
            if self.enable_adversary_experts
            else False
        )
        self.activate_adv(activate_flag)
        image_embeddings = self.sam.image_encoder(images)
        cls_prediction = self.detector(image_embeddings)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        device = image_embeddings.device
        mask_prediction, iou_prediction = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe().to(device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # (1, 1, H//4, W//4) (1, 1)
        return mask_prediction, cls_prediction

    # === Load separately ===
    def load_sam_lora_parameters(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self._load_state_dict(self.sam, state_dict)

    def load_detector_parameters(self, path):
        if self.detector is None:
            raise ValueError("Detector is not defined.")
        state_dict = torch.load(path, map_location="cpu")
        self._load_state_dict(self.detector, state_dict)

    # === Save all trainable parameters at once ===
    def save_all_parameters(self, path):
        all_params = {}
        all_params["sam"] = {
            k: v for k, v in self._get_named_params(self.sam) if v.requires_grad
        }
        if self.detector is not None:
            all_params["detector"] = {
                k: v
                for k, v in self._get_named_params(self.detector)
                if v.requires_grad
            }
        torch.save(all_params, path)

    # === Load all at once ===
    def load_all_parameters(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pretrained parameter file not found: {path}")
        all_params = torch.load(path, map_location="cpu")
        print(path)
        if "sam" in all_params:
            self._load_state_dict(self.sam, all_params["sam"])
        if "detector" in all_params and self.detector is not None:
            self._load_state_dict(self.detector, all_params["detector"])

    # === Utility functions ===
    def _get_named_params(self, module):
        if isinstance(
            module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            return module.module.named_parameters()
        return module.named_parameters()

    def _load_state_dict(self, module, state_dict):
        if isinstance(
            module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            module.module.load_state_dict(state_dict, strict=False)
        else:
            module.load_state_dict(state_dict, strict=False)


model_type = ["vit_b", "vit_l", "vit_h"]
checkpoint = {
    "vit_b": "./weight/pre-trained_weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/pre-trained_weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/pre-trained_weight/sam_vit_h_4b8939.pth",
}
