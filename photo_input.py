import streamlit as st
import cv2
from PIL import Image
import time
import os
import pygame
from ultralytics import YOLO

st.set_page_config(page_title="💵 Currency Detection", layout="wide")
st.title("📸 Currency Detection (Live Stream + YOLO on Capture)")

pygame.mixer.init()

if "captured" not in st.session_state:
    st.session_state.captured = False

if "start_played" not in st.session_state:
    st.session_state.start_played = False

start_audio = "audios/start.mp3"
if not st.session_state.start_played and os.path.exists(start_audio):
    try:
        pygame.mixer.music.load(start_audio)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        st.session_state.start_played = True
    except Exception as e:
        st.error(f"❌ Failed to play start.mp3: {e}")
elif not os.path.exists(start_audio):
    st.warning("⚠️ Missing start.mp3 in audios folder.")

model = YOLO("runs/detect/train9/weights/best.pt")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
live_col, captured_col = st.columns(2)
live_placeholder = live_col.empty()
captured_placeholder = captured_col.empty()
captured_col.markdown("### 🖼️ Last Captured Image")

if st.button("📸 Capture Photo"):
    st.session_state.captured = True

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Webcam not found!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        live_placeholder.image(rgb_frame, caption="📡 Live Feed", channels="RGB")

        if st.session_state.captured:
            filename = f"capimages/captured_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"✅ Captured: {filename}")

            results = model(frame, imgsz=640, conf=conf_threshold)
            annotated_frame = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            captured_img = Image.fromarray(annotated_rgb)
            captured_placeholder.image(captured_img, caption="🖼️ Detected Image", channels="RGB")

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    audio_path = f"audios/{label.upper()}.mp3"
                    st.write(f"🪙 Detected: {label}")

                    if os.path.exists(audio_path):
                        try:
                            pygame.mixer.music.load(audio_path)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                        except Exception as e:
                            st.error(f"🎧 Error playing audio: {e}")
                    else:
                        st.warning(f"⚠️ Missing audio: {audio_path}")

            st.session_state.captured = False

        time.sleep(0.03)

cap.release()
