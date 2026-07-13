import torch
import torchvision.models as models
from models import cbam_resnext as customized_models
from torch import nn

# Models
default_model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

customized_models_names = sorted(
    name
    for name in customized_models.__dict__
    if not name.startswith("__") and callable(customized_models.__dict__[name])
)

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


def make_model(model_name, num_classes):
    print("creating model '{}'".format(model_name))
    model = models.__dict__[model_name](progress=True)

    model.fc = nn.Sequential(nn.Linear(2048, num_classes))
    return model


if __name__ == "__main__":
    model = make_model("resnext101_32x8d_swsl", 43)
    check = torch.load("swsl_8_cbam2_24.pth")
    model.load_state_dict(check)
    x = torch.randn((1, 3, 288, 288))
    print(model(x))
    # all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    # print(all_model)
