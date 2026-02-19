import os
import gdown
import streamlit as st

MODEL_PATH = "final_best_model.pth"

# Paste your File ID here
FILE_ID = "1vYNgfefvy7XW_bNb1A6mxiyxpbgDNlgS"

# Direct download link
url = f"https://drive.google.com/file/d/1vYNgfefvy7XW_bNb1A6mxiyxpbgDNlgS/view?usp=sharing"

# Download model if not already downloaded
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model from Google Drive... Please wait ⏳")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully ✅")

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Hybrid Quantum Breast Cancer Detector",
    page_icon="??",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "final_best_model.pth"
model = load_model(MODEL_PATH)

class_names = ["benign", "malignant", "normal"]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# -----------------------------
# Grad-CAM Function
# -----------------------------
def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.resnet.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    grad = gradients[0].detach().numpy()
    act = activations[0].detach().numpy()

    weights = np.mean(grad, axis=(2, 3))
    cam = np.zeros(act.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()
    return cam

# -----------------------------
# UI Header
# -----------------------------
st.title("?? Hybrid Quantum-Classical Breast Cancer Detection")
st.markdown("### ResNet50 + Variational Quantum Classifier (VQC) + Grad-CAM")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred_class].item()

    st.success(f"Prediction: **{class_names[pred_class].upper()}**")
    st.info(f"Confidence Score: **{confidence:.2f}**")

    # -----------------------------
    # Feature 1: Probability Chart
    # -----------------------------
    st.subheader("?? Class Probability Distribution")
    prob_values = probs[0].numpy()

    st.bar_chart({
        "benign": prob_values[0],
        "malignant": prob_values[1],
        "normal": prob_values[2]
    })

    # -----------------------------
    # Grad-CAM Heatmap
    # -----------------------------
    st.subheader("?? Grad-CAM Tumor Region Highlight")

    cam = generate_gradcam(model, input_tensor, pred_class)

    img_np = np.array(img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * 0.4 + img_np

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_np, caption="Original", clamp=True)

    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap", clamp=True)

    with col3:
        st.image(overlay, caption="Overlay Output", clamp=True)

    # -----------------------------
    # Feature 2: Edge Detection
    # -----------------------------
    st.subheader("?? Edge Pattern Detection")

    gray = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    st.image(edges, caption="Canny Edge Detection Output")

    # -----------------------------
    # Feature 3: Tumor Activation Score
    # -----------------------------
    st.subheader("?? Tumor Activation Score")

    activation_score = np.mean(cam)
    st.write("Tumor Activation Score:", round(float(activation_score), 4))

    # -----------------------------
    # Download Diagnosis Report
    # -----------------------------
    st.subheader("?? Download Report")

    report_text = f"""
    Breast Cancer Detection Report
    -----------------------------
    Prediction: {class_names[pred_class]}
    Confidence: {confidence:.2f}

    Tumor Activation Score: {activation_score:.4f}
    """

    st.download_button(
        label="Download Report as TXT",
        data=report_text,
        file_name="diagnosis_report.txt"

    )

