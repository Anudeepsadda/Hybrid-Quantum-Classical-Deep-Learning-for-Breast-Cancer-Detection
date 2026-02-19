import torch
import torch.nn as nn
from torchvision import models


# ==========================================================
# ResNet50 Breast Cancer Classifier (3 Classes)
# ==========================================================
class ResNetBreastCancer(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ==========================================================
# Load Model Function
# ==========================================================
def load_model(weight_path):

    model = ResNetBreastCancer(num_classes=3)

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model
