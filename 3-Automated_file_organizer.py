import os
import shutil
import random


base_dir = 'currency_dataset'
yolo_dir = 'currency_dataset_yolo'
images_train = os.path.join(yolo_dir, 'images', 'train')
images_val = os.path.join(yolo_dir, 'images', 'val')
labels_train = os.path.join(yolo_dir, 'labels', 'train')
labels_val = os.path.join(yolo_dir, 'labels', 'val')

os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

classes = ['20RUPEES', '50RUPEES', '100RUPEES', '500RUPEES', '1000RUPEES', '5000RUPEES']

# Split: 80% train, 20% val
for cls in classes:
    img_folder = os.path.join(base_dir, cls, 'images')
    label_folder = os.path.join(base_dir, cls, 'labels')

    files = sorted(os.listdir(img_folder))
    random.shuffle(files)
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for fname in train_files:
        shutil.copy(os.path.join(img_folder, fname), images_train)
        shutil.copy(os.path.join(label_folder, fname.replace('.jpg', '.txt')), labels_train)

    for fname in val_files:
        shutil.copy(os.path.join(img_folder, fname), images_val)
        shutil.copy(os.path.join(label_folder, fname.replace('.jpg', '.txt')), labels_val)
