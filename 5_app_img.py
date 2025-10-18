import streamlit as st
import requests
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import io
import time
import numpy as np

# Constants
ESP32_IP = "http://192.168.137.28"
CAPTURE_ENDPOINT = f"{ESP32_IP}/capture"
RESULT_ENDPOINT = f"{ESP32_IP}/result"
STATUS_ENDPOINT = f"{ESP32_IP}/status"
MODEL_PATH = "runs/detect/train9/weights/best.pt"  # train3(100img) - train6(150img) - train9(170img)

model = YOLO(MODEL_PATH)
model.fuse()
model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

def detect_currency_yolo(image_bytes, conf_threshold):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(img, imgsz=640, conf=conf_threshold)

        if results and len(results[0].boxes) > 0:
            names = model.names
            boxes = results[0].boxes
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            top_idx = np.argmax(scores)
            top_score = scores[top_idx]

            if top_score < conf_threshold:
                return None, None

            detected_class = names.get(class_ids[top_idx], f"Class_{class_ids[top_idx]}")
            annotated_img = results[0].plot()
            return detected_class, annotated_img
    except Exception as e:
        st.error(f"⚠️ Detection failed: {e}")
    return None, None

def wait_for_audio_finish(timeout=10):
    start = time.time()
    with st.spinner("🔊 Waiting for ESP32 to finish speaking..."):
        while True:
            try:
                r = requests.get(STATUS_ENDPOINT, timeout=3)
                if r.status_code == 200 and r.text.strip().lower() == "idle":
                    return True
            except:
                pass
            if time.time() - start > timeout:
                st.warning("⚠️ Timeout waiting for ESP32 audio.")
                return False
            time.sleep(0.3)

if "image_id_counter" not in st.session_state:
    st.session_state.image_id_counter = 1

st.set_page_config(page_title="YOLO Currency Detector", layout="centered")
st.title("💵 ESP32 Currency Detector (Live-like)")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

if st.button("🎥 Capture from ESP32 and Detect"):
    try:
        image_id = f"img_{st.session_state.image_id_counter}"
        st.session_state.image_id_counter += 1

        with st.spinner(f"📡 Capturing frame from ESP32 with ID: {image_id}..."):
            response = requests.get(f"{CAPTURE_ENDPOINT}?id={image_id}", timeout=10)

        if response.status_code != 200:
            st.error("❌ Failed to capture image from ESP32.")
        else:
            confirmed_id = response.headers.get("X-Capture-ID", "unknown")
            image_bytes = response.content
            raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            label, annotated_image = detect_currency_yolo(image_bytes, conf_threshold)

            st.info(f"🆔 Image ID: `{confirmed_id}`")

            col1, col2 = st.columns(2)
            with col1:
                st.image(raw_img, caption="📷 Raw Image", use_container_width=True)
            with col2:
                if annotated_image is not None:
                    st.image(annotated_image, caption=f"🧠 Prediction: {label}", use_container_width=True)
                    st.success(f"🟢 **Detected Currency:** {label}")

                    # Trigger ESP32 audio output
                    with st.spinner(f"🔊 Triggering ESP32 audio for '{label}'..."):
                        audio_resp = requests.get(f"{RESULT_ENDPOINT}?value={label}", timeout=5)
                        if audio_resp.status_code == 200:
                            wait_for_audio_finish()
                        else:
                            st.warning("⚠️ ESP32 did not accept audio trigger.")
                else:
                    st.warning("⚠️ No currency detected.")
    except Exception as e:
        st.error(f"🚫 Error: {e}")
