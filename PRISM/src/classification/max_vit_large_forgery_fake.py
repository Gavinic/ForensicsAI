import os

import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
from config.default import cfg
from engine import trainer_normal
from torch import nn
from utils.compare import compare, count
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from utils.seed import set_seed

cfg.merge_from_file("config/max_vit_large_forgery_fake.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(
    ",".join(str(i) for i in cfg.SYSTEM.GPU_ID)
)
torch.cuda.empty_cache()

model = timm.create_model("maxvit_large_tf_512.in21k_ft_in1k", pretrained=False)
print(model)
weight_path = "<BASE_PATH>/ckpts/maxvit_large_tf_512.in21k_ft_in1k.bin"
state_dict = torch.load(weight_path, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
num_ftrs = model.head.fc.in_features
model.head.fc = nn.Linear(num_ftrs, 2)
print(model)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
# state_dict = torch.load(weight_path, map_location="cpu")
# model.load_state_dict(state_dict, strict=True)
# print(model)


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum() if reduction == "sum" else loss
    )


def lin_comb(a, b, epsilon):
    return epsilon * a + b * (1 - epsilon)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean"):
        super().__init__()
        self.epsilon, self.reduction = epsilon, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss / c, nll, self.epsilon)


model = nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss().cuda()
# criterion = LabelSmoothingCrossEntropy().cuda()
optimizer = optim.SGD((model.parameters()), lr=0.0002, momentum=0.9, weight_decay=1e-4)
trainer_engine = trainer_normal.BASE(cfg)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20, eta_min=8e-6
)
trainer_engine.train_model(model, criterion, optimizer, lr_scheduler)
torch.cuda.empty_cache()
