import torch
import torch.nn as nn
from torchvision import models


# ==========================================================
# RESNET50 CLASSIFIER
# ==========================================================
class ResNetBreastCancer(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.resnet = models.resnet50(weights=None)

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)


# ==========================================================
# LOAD MODEL SAFELY (IGNORE QUANTUM KEYS)
# ==========================================================
def load_model(weight_path):

    model = ResNetBreastCancer(num_classes=3)

    # Load checkpoint dictionary
    checkpoint = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=False
    )

    # Remove unwanted quantum layer weights automatically
    filtered_checkpoint = {}

    for key, value in checkpoint.items():
        if "resnet" in key:   # only keep ResNet weights
            filtered_checkpoint[key] = value

    # Load only ResNet weights
    model.load_state_dict(filtered_checkpoint, strict=False)

    model.eval()
    return model
