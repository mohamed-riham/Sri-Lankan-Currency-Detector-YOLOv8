import cv2
import os

classes = ['20RUPEES', '50RUPEES', '100RUPEES', '500RUPEES', '1000RUPEES', '5000RUPEES']
start_index = 150
num_images_to_add = 20  # can change this to 100 or 200
output_dir = 'currency_dataset'
img_size = (640, 480)

for cls in classes:
    os.makedirs(os.path.join(output_dir, cls, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, cls, 'labels'), exist_ok=True)

cap = cv2.VideoCapture(1)

for cls_id, cls in enumerate(classes):
    count = start_index
    end_index = start_index + num_images_to_add
    print(f"\n📸 Capturing for {cls}... Press SPACE to capture, Q to skip.")

    while count < end_index:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        cv2.putText(frame, f"{cls}: {count}/{end_index - 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            img_name = f"{cls}_{count:03d}.jpg"
            img_path = os.path.join(output_dir, cls, 'images', img_name)
            cv2.imwrite(img_path, frame)
            print(f"✅ Saved: {img_name}")
            count += 1

cap.release()
cv2.destroyAllWindows()

print("\n✏️ Labeling phase started. Draw boxes with mouse, press ENTER to save, ESC to skip.")

def draw_manual_bbox(image_path, label_path, class_id):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    clone = img.copy()
    bbox = []

    def click_and_drag(event, x, y, flags, param):
        nonlocal bbox
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            bbox.append((x, y))
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.imshow("Label", img)

    cv2.namedWindow("Label")
    cv2.setMouseCallback("Label", click_and_drag)

    while True:
        cv2.imshow("Label", img)
        key = cv2.waitKey(0) & 0xFF
        if key == 13 and len(bbox) == 2:  # ENTER key
            (x1, y1), (x2, y2) = bbox
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cx = ((x1 + x2) / 2) / img_size[0]
            cy = ((y1 + y2) / 2) / img_size[1]
            w = (x2 - x1) / img_size[0]
            h = (y2 - y1) / img_size[1]
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            print(f"✅ Labeled and saved: {os.path.basename(label_path)}")
            break
        elif key == 27:  # ESC
            print("⏭️ Skipped image.")
            break
        elif key == ord('r'):
            img[:] = clone.copy()
            bbox = []
            print("🔁 Reset.")

    cv2.destroyWindow("Label")

for cls_id, cls in enumerate(classes):
    img_folder = os.path.join(output_dir, cls, 'images')
    label_folder = os.path.join(output_dir, cls, 'labels')
    images = sorted(os.listdir(img_folder))

    for img_file in images:
        try:
            index = int(img_file.split('_')[-1].split('.')[0])
        except:
            continue
        if index < start_index:
            continue

        img_path = os.path.join(img_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace('.jpg', '.txt'))
        draw_manual_bbox(img_path, label_path, cls_id)

print("\n✅ All new images labeled.")
