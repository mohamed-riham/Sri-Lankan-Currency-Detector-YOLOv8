# 🇱🇰 Sri Lankan Currency Detector (YOLOv8)

An advanced real-time object detection system to identify and classify **Sri Lankan Rupee (LKR)** banknotes using the **YOLOv8** model.  
This project combines **computer vision**, **deep learning**, and **automation** to assist with visual currency recognition.

---

[![Demo Video](VisionAid/thumbnail.jpg)](VisionAid/demo.mp4)


---

## 💰 Detected Denominations

- **Rs. 20**
- **Rs. 50**
- **Rs. 100**
- **Rs. 500**
- **Rs. 1000**
- **Rs. 5000**

---

## 💡 Key Features

- 🧠 **Model:** Utilizes the lightweight and efficient **YOLOv8n** model for fast inference.  
- 🎥 **Detection Modes:** Supports detection on static **images**, **live video streams** (webcam), and **pre-recorded videos**.  
- 📸 **Comprehensive Dataset:** Trained on a **custom dataset of 1000+ images** of Sri Lankan currency notes.  
- ⚡ **Performance:** Real-time inference optimized for both CPU and GPU environments.

---

## ⚙️ Project Structure

| Folder/File | Description |
|--------------|-------------|
| `VisionAid/` | Contains main source code, training scripts, and utilities. |
| `VisionAid/4-train.py` | Script to train the YOLOv8 model. |
| `VisionAid/5-app_img.py` | Application script for image-based detection. |
| `VisionAid/app_live_capture.py` | Real-time webcam or video stream detection. |
| `VisionAid/currency_dataset/` | Dataset folder containing training images and labels. |
| `VisionAid/runs/` | Stores trained model weights (`.pt`) and training logs. |
| `Project/` | Includes additional project scripts (e.g., HTTP detection server). |
| `currency_http_detect.py` | Simple HTTP server for detection via an API. |
| `run.bat` | Windows batch file for quickly launching the main app. |

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Ensure you have **Python 3.8+** installed.

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/YourUsername/sri-lankan-currency-detector-yolov8.git
cd sri-lankan-currency-detector-yolov8
```

---

### 3️⃣ Install Dependencies

It’s recommended to create a virtual environment before installation.

```bash
# Install required libraries
pip install ultralytics opencv-python pillow
# Or install all dependencies from a file
# pip install -r requirements.txt
```

---

### 4️⃣ Setup the Dataset and Models

**Dataset:**  
Extract the dataset into the directory:
```
VisionAid/currency_dataset/
```

**Trained Model:**  
Place the YOLOv8 weight file (e.g., `best.pt`) in:
```
VisionAid/runs/detect/train/weights/
```

---

## 💻 Usage Examples

### 🖼️ A. Detect on a Static Image
```bash
python VisionAid/5-app_img.py --source /path/to/your/image.jpg
```

### 🎥 B. Live Webcam Detection
```bash
python VisionAid/app_live_capture.py
```

### 🎬 C. Detect on a Video File
```bash
python VisionAid/app_live_capture.py --source /path/to/your/video.mp4
```

---

## 🛠️ Training the Model (Optional)

If you want to **retrain** or fine-tune the model:

1. Organize your dataset in YOLO format under:
   ```
   VisionAid/currency_dataset/
   ```
2. Adjust training parameters in:
   ```
   VisionAid/4-train.py
   ```
3. Run the training command:
   ```bash
   python VisionAid/4-train.py
   ```

The new model weights will be saved inside:
```
VisionAid/runs/detect/
```

---

## 🤝 Contribution

Contributions are always welcome!  
Feel free to open **issues** or **pull requests** for improvements, bug fixes, or feature additions.

---

## 🧑‍💻 Author

**Mohamed Riham**  
Software Engineer | AI Developer | Data Science Student  
📍 Sri Lanka  
🔗 [LinkedIn](https://linkedin.com/in/mohamedriham) • [GitHub](https://github.com/mohamed-riham)

---

## 🪄 Future Enhancements

- Add multilingual audio feedback (via DFPlayer Mini).  
- Integrate ESP32-CAM live capture stream.  
- Enable offline model quantization for edge deployment.

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.
