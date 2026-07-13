import torch
import torch.fft
import torch.nn as nn
import torchvision.models as models

try:
    import timm
except ImportError:
    timm = None


class AdversaryDetector(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "convnext_tiny":
            if timm is None:
                raise ImportError(
                    "timm is required for backbone='convnext_tiny'. Please install timm or use resnet backbones."
                )
            self.backbone = timm.create_model(
                "convnext_tiny", pretrained=True, num_classes=0, global_pool="avg"
            )
            in_features = self.backbone.num_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")

        # Project to a 256-dim contrastive learning space (intermediate features)
        self.feat_proj = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU())

        # Final classifier (normal vs adversarial)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        # Feature extraction
        raw_features = self.backbone(x)  # shape [B, in_features]
        features = self.feat_proj(raw_features)  # shape [B, 256]
        logits = self.classifier(features)  # shape [B, 2]

        return logits, features

    def save_detector(self, path):
        assert isinstance(self, nn.Module), "model must be an nn.Module instance"
        torch.save({k: v for k, v in self.state_dict().items()}, path)

    def load_detector(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(
            self, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ):
            self.module.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)
