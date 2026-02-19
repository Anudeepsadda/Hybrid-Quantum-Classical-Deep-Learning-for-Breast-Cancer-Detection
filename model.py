import torch
import torch.nn as nn
from torchvision import models


# ============================================
# ResNet50 Breast Cancer Classifier (3 Classes)
# ============================================
class BreastCancerResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Load ResNet50 Backbone
        self.resnet = models.resnet50(weights=None)

        # Replace final FC layer for 3-class classification
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)


# ============================================
# Load Model Function (Streamlit Compatible)
# ============================================
def load_model(model_path):

    model = BreastCancerResNet(num_classes=3)

    # ✅ Fix: Load GPU-trained model safely on CPU
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu")
    )

    model.load_state_dict(state_dict)

    model.eval()
    return model
