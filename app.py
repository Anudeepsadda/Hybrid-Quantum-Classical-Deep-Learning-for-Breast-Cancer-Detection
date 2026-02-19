import os
import gdown
import streamlit as st

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt

from model import load_model


# ==========================================================
# STREAMLIT PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Hybrid Quantum Breast Cancer Detector",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Hybrid Quantum-Classical Breast Cancer Detection")
st.markdown("### ResNet50 + Variational Quantum Classifier (VQC) + Grad-CAM")


# ==========================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ==========================================================

MODEL_PATH = "best_model.pth"

FILE_ID = "15Ak5r05RLhr-6v77J5DyOoBceov0n54n"

# ✅ Correct direct download link
url = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model from Google Drive... ⏳")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully ✅")

# ==========================================================
# LOAD MODEL
# ==========================================================
st.info("🔄 Loading Hybrid Quantum Model...")

model = load_model(MODEL_PATH)
model.eval()

st.success("✅ Model Loaded Successfully!")


# ==========================================================
# CLASS LABELS
# ==========================================================
class_names = ["benign", "malignant", "normal"]


# ==========================================================
# IMAGE TRANSFORMS
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


# ==========================================================
# GRAD-CAM FUNCTION
# ==========================================================
def generate_gradcam(model, input_tensor, target_class):

    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target layer = ResNet last block
    target_layer = model.resnet.layer4

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    loss = output[0, target_class]

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Extract gradients + activations
    grad = gradients[0].detach().numpy()
    act = activations[0].detach().numpy()

    # Compute Grad-CAM heatmap
    weights = np.mean(grad, axis=(2, 3))
    cam = np.zeros(act.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    return cam


# ==========================================================
# UPLOAD IMAGE UI
# ==========================================================
st.subheader("📌 Upload Breast Ultrasound Image")

uploaded_file = st.file_uploader(
    "Upload an ultrasound image (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)


# ==========================================================
# MAIN PREDICTION PIPELINE
# ==========================================================
if uploaded_file:

    # Load Image
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

    # Display Prediction
    st.success(f"✅ Prediction: **{class_names[pred_class].upper()}**")
    st.info(f"🎯 Confidence Score: **{confidence:.2f}**")

    # ==========================================================
    # FEATURE 1: CLASS PROBABILITY DISTRIBUTION
    # ==========================================================
    st.subheader("📊 Class Probability Distribution")

    prob_values = probs[0].numpy()

    st.bar_chart({
        "Benign": prob_values[0],
        "Malignant": prob_values[1],
        "Normal": prob_values[2]
    })

    # ==========================================================
    # FEATURE 2: GRAD-CAM HEATMAP
    # ==========================================================
    st.subheader("🔥 Grad-CAM Tumor Region Visualization")

    cam = generate_gradcam(model, input_tensor, pred_class)

    # Convert original image to numpy
    img_np = np.array(img.resize((224, 224))) / 255.0

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Overlay
    overlay = heatmap * 0.4 + img_np

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_np, caption="Original Image", clamp=True)

    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap", clamp=True)

    with col3:
        st.image(overlay, caption="Overlay Output", clamp=True)

    # ==========================================================
    # FEATURE 3: EDGE PATTERN DETECTION
    # ==========================================================
    st.subheader("🧩 Edge Pattern Detection (Tumor Boundary)")

    gray = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    st.image(edges, caption="Canny Edge Detection Output")

    # ==========================================================
    # FEATURE 4: TUMOR ACTIVATION SCORE
    # ==========================================================
    st.subheader("📌 Tumor Activation Score")

    activation_score = np.mean(cam)

    st.write("Average Activation Score:", round(float(activation_score), 4))

    # ==========================================================
    # FEATURE 5: DOWNLOAD REPORT
    # ==========================================================
    st.subheader("📄 Download Diagnosis Report")

    report_text = f"""
    Hybrid Quantum Breast Cancer Detection Report
    --------------------------------------------

    Prediction Class : {class_names[pred_class]}
    Confidence Score : {confidence:.2f}

    Tumor Activation Score : {activation_score:.4f}

    Model Used : ResNet50 + Variational Quantum Classifier (VQC)
    Dataset   : BUSI Breast Ultrasound Dataset
    """

    st.download_button(
        label="⬇️ Download Report as TXT",
        data=report_text,
        file_name="diagnosis_report.txt"
    )


