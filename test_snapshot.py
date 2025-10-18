import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLOv8 Currency Detection")

# Load YOLOv8 model
model = YOLO("runs/detect/train9/weights/best.pt")

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)

# Camera input from phone browser
img_data = st.camera_input("Take a photo using your phone camera")

# Process if image is captured
if img_data:
    image = Image.open(img_data)
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np, imgsz=640, conf=conf_threshold)
    annotated = results[0].plot()

    # Display result
    st.image(annotated, caption="Detection Result", channels="BGR")
