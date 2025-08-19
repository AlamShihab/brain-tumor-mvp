# app/app.py
import streamlit as st
from PIL import Image
from src.inference.predict import load_model, predict
# Optional Grad-CAM
from src.inference.gradcam import GradCAM, overlay_cam
import torch

st.set_page_config(page_title="Brain Tumor Classifier MVP")
st.title("Brain Tumor Classifier — MVP")
st.caption("Research/educational use only — not medical advice.")

# Load model
model = load_model()

# Grad-CAM setup
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI", use_column_width=True)
    
    # Prediction
    result = predict(uploaded_file, model)
    st.success(f"Predicted class: **{result['class']}**")
    st.write("Class probabilities:", result["probs"])
    
    # Grad-CAM overlay
    x = transform(img).unsqueeze(0).to(device)
    cam = gradcam.generate_cam(x, class_idx=list(result["probs"]).index(max(result["probs"])))
    overlay = overlay_cam(img, cam)
    st.subheader("Grad-CAM Heatmap")
    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
