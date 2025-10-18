from ultralytics import YOLO

model = YOLO("currency_dataset/yolov8n.pt")  # yolov8s.pt, yolov8m.pt
model.train(data="currency_dataset_yolo/data.yaml", epochs=50, imgsz=640)
