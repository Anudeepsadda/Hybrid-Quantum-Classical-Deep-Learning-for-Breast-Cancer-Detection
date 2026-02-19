import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml


# -----------------------------
# Quantum Layer (VQC)
# -----------------------------
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class VQCLayer(nn.Module):
    def __init__(self):
        super().__init__()

        weight_shapes = {"weights": (2, n_qubits, 3)}

        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

    def forward(self, x):
        return self.qlayer(x)


# -----------------------------
# Hybrid Quantum ResNet Model
# -----------------------------
class HybridQuantumResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ✅ ResNet50 Backbone
        self.resnet = models.resnet50(weights=None)

        # Remove final FC
        self.resnet.fc = nn.Identity()

        # Classical Layer before Quantum
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits)
        )

        # Quantum Layer
        self.vqc = VQCLayer()

        # Final Output Layer
        self.fc2 = nn.Sequential(
            nn.Linear(n_qubits, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.vqc(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Load Model Correctly
# -----------------------------
def load_model(model_path):
    model = HybridQuantumResNet(num_classes=3)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # If checkpoint contains extra dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # ✅ Fix key mismatch automatically
    new_state_dict = {}

    for key, value in checkpoint.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)

    model.eval()
    return model
