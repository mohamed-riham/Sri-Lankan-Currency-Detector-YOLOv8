from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train9/weights/best.pt")  # train3(100img) - train6(150img) - train9(170img)

input_path = "test_vid/test.mp4"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("❌ Failed to open the video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
output_path = "outputvids/output_video_150imgs.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.5)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output video saved to: {output_path}")
