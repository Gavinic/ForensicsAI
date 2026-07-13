import sys

import torch
import torch.nn as nn

from .vit_library.segment_anything import sam_model_registry


class FOCAL_ViT(nn.Module):
    def __init__(self, checkpoint=None, model_type="vit_l"):
        super().__init__()
        self.name = "FOCAL_ViT"
        self.net = sam_model_registry[model_type](checkpoint=checkpoint)

    def forward(self, x):
        x = self.net.image_encoder(x)
        return x
