import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
from pennylane.qnn import TorchLayer

# -----------------------------
# Quantum Circuit Setup
# -----------------------------
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = TorchLayer(quantum_circuit, weight_shapes)

# -----------------------------
# Hybrid ResNet + VQC Model
# -----------------------------
class HybridResNetVQC(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.fc1 = nn.Linear(2048, 4)
        self.vqc = qlayer
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.vqc(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Load Model Function
# -----------------------------
def load_model(weight_path):
    model = HybridResNetVQC()
    model.load_state_dict(
        torch.load(weight_path, map_location="cpu")
    )
    model.eval()

    return model

