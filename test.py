import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import time

st.title("YOLOv8 Currency Detection with Threshold")

model = YOLO("runs/detect/train9/weights/best.pt")  # train3(100img) - train6(150img) - train9(170img)

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)

frame_placeholder = st.empty()

# Start video capture directly on app start
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Failed to read from webcam.")
            break

        results = model(frame, imgsz=640, conf=conf_threshold)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame)

        frame_placeholder.image(img, channels="RGB")

        # short delay to reduce CPU usage and allow UI update
        time.sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
