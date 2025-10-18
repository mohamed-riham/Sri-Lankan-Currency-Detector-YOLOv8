import streamlit as st
import requests
from PIL import Image
from ultralytics import YOLO
import io
import time
import numpy as np
import os
import pygame
import cv2

# Constants
ESP32_IP = "http://192.168.137.28"
CAPTURE_ENDPOINT = f"{ESP32_IP}/capture"
MODEL_PATH = "runs/detect/train9/weights/best.pt"  # Update with your best model path
AUDIO_DIR = "audios"
NUM_PHOTOS = 5
WAIT_TIME_BETWEEN_CAPTURES = 5  # seconds


@st.cache_resource(ttl=3600)
def load_model():
    model = YOLO(MODEL_PATH)
    model.fuse()
    # Warmup (run a dummy prediction)
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    return model


model = load_model()


def init_audio():
    try:
        pygame.mixer.init()
    except pygame.error:
        pass


init_audio()


def detect_currency_yolo(image_bytes, conf_threshold):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(img, imgsz=640, conf=conf_threshold)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            top_idx = np.argmax(scores)
            top_score = scores[top_idx]

            if top_score < conf_threshold:
                return None, None

            detected_class = model.names.get(class_ids[top_idx], f"Class_{class_ids[top_idx]}")

            annotated_img = results[0].plot()
            # Convert BGR (OpenCV default) to RGB for Streamlit
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            return detected_class, annotated_img

    except Exception as e:
        st.error(f"⚠ Detection failed: {e}")

    return None, None


# Streamlit UI setup
st.set_page_config(page_title="YOLO Currency Detector", layout="centered")
st.title("💵 Currency Detector (Local Audio)")

with st.sidebar:
    st.header("⚙ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

if "image_id_counter" not in st.session_state:
    st.session_state.image_id_counter = 1


def play_audio(label):
    audio_path = os.path.join(AUDIO_DIR, f"{label}.mp3")
    if os.path.exists(audio_path):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except Exception as e:
            st.warning(f"🔇 Failed to play audio: {e}")
    else:
        st.warning(f"🔇 Audio file for '{label}' not found in {AUDIO_DIR}/")


if st.button(f"🎥 Capture {NUM_PHOTOS} photos from ESP32 and Detect"):
    placeholder = st.empty()  # this clears previous content each time

    for i in range(NUM_PHOTOS):
        image_id = f"img_{st.session_state.image_id_counter}"
        st.session_state.image_id_counter += 1

        with placeholder.container():  # This clears previous images automatically
            with st.spinner(f"📡 Capturing frame {i + 1}/{NUM_PHOTOS} from ESP32 with ID: {image_id}..."):
                try:
                    response = requests.get(f"{CAPTURE_ENDPOINT}?id={image_id}", timeout=100)
                    response.raise_for_status()
                except requests.RequestException as e:
                    st.error(f"❌ Failed to capture image {i + 1} from ESP32: {e}")
                    continue

            confirmed_id = response.headers.get("X-Capture-ID", "unknown")
            image_bytes = response.content

            raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            label, annotated_image = detect_currency_yolo(image_bytes, conf_threshold)

            st.info(f"🆔 Image ID: {confirmed_id} (Capture {i + 1}/{NUM_PHOTOS})")

            col1, col2 = st.columns(2)
            with col1:
                st.image(raw_img, caption=f"📷 Raw Image {i + 1}", use_container_width=True)

            with col2:
                if annotated_image is not None:
                    st.image(annotated_image, caption=f"🧠 Prediction: {label}", use_container_width=True)
                    st.success(f"🟢 Detected Currency: {label}")
                    play_audio(label)
                else:
                    st.warning("⚠ No currency detected.")

        if i < NUM_PHOTOS - 1:
            time.sleep(WAIT_TIME_BETWEEN_CAPTURES)
