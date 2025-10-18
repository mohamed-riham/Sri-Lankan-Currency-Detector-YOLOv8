import streamlit as st
import requests
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np

ESP32_IP = "http://192.168.137.28"
CAPTURE_URL = f"{ESP32_IP}/capture"
STREAM_URL = f"{ESP32_IP}:81/stream"
MODEL_PATH = "runs/detect/train6/weights/best.pt"   # train3(100img) - train6(150img) - train9(170img)
IMAGE_PATH = "test.jpg"

model = YOLO(MODEL_PATH)

if 'history' not in st.session_state:
    st.session_state.history = []

def detect_currency_yolo(image_path, conf_threshold):
    results = model(image_path, imgsz=640, conf=conf_threshold)
    if results and len(results[0].boxes) > 0:
        names = model.names
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_classes = [names[cid] for cid in class_ids]
        return detected_classes[0], results[0].plot()
    return None, None

st.set_page_config(page_title="YOLO Currency Detector", layout="centered")
st.title("💵 ESP32 Currency Detector with YOLOv8")

with st.sidebar:
    st.header("⚙️ Detection Settings")
    mode = st.radio("Mode", ["📸 Capture & Detect", "📺 Live Stream"])
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    show_raw_image = st.checkbox("Show Captured Raw Image", value=False)

if mode == "📸 Capture & Detect":
    if st.button("Capture and Detect Currency"):
        with st.spinner("Capturing image from ESP32..."):
            try:
                response = requests.get(CAPTURE_URL, timeout=10)
                if response.status_code == 200:
                    with open(IMAGE_PATH, "wb") as f:
                        f.write(response.content)
                    st.success("✅ Image captured successfully!")

                    if show_raw_image:
                        st.subheader("🖼️ Raw Captured Image")
                        raw_image = Image.open(IMAGE_PATH)
                        st.image(raw_image, caption="Raw Photo", use_column_width=True)

                    label, annotated_image = detect_currency_yolo(IMAGE_PATH, conf_threshold)

                    if label:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.subheader("🔍 Detection Result")
                        st.image(annotated_image, caption=f"Detected: {label}", use_container_width=True)
                        st.info(f"🟢 **Currency Detected:** {label}")

                        st.session_state.history.append((timestamp, label))

                        requests.get(f"{ESP32_IP}/result?value={label}")

                        with open(IMAGE_PATH, "rb") as file:
                            st.download_button(
                                label="💾 Download Captured Image",
                                data=file,
                                file_name="currency_detected.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.warning("⚠️ No currency detected.")
                else:
                    st.error("❌ Failed to get a valid response from ESP32.")
            except Exception as e:
                st.error(f"🚫 Error: {e}")

elif mode == "📺 Live Stream":
    st.subheader("Live Stream from ESP32-CAM")
    frame_window = st.empty()
    try:
        stream = requests.get(STREAM_URL, stream=True, timeout=10)
        bytes_data = b""
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_column_width=True)
    except Exception as e:
        st.error(f"Live stream error: {e}")

# Detection history
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Detection History")
    for timestamp, detected in reversed(st.session_state.history[-5:]):
        st.write(f"🕒 {timestamp} — 🏷️ **{detected}**")
