import torch
import torch.nn as nn
from torchvision import models


# ==========================================================
# RESNET50 BREAST CANCER CLASSIFIER (3-Class)
# ==========================================================
class ResNetBreastCancer(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetBreastCancer, self).__init__()

        # Load ResNet50 backbone (no pretrained weights for deployment)
        self.resnet = models.resnet50(weights=None)

        # Replace Final Fully Connected Layer for 3 Classes
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)


# ==========================================================
# LOAD MODEL FUNCTION
# ==========================================================
def load_model(weight_path):
    """
    Loads the ResNet50 breast cancer classifier model safely.
    Compatible with Streamlit Cloud Python 3.13+
    """

    # Create Model Architecture
    model = ResNetBreastCancer(num_classes=3)

    # Load Weights Safely
    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=False   # IMPORTANT FIX for Streamlit + Torch 2.6+
    )

    model.load_state_dict(state_dict)

    # Set to Evaluation Mode
    model.eval()

    return model
