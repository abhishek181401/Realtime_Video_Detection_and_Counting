import os
import urllib.request

os.makedirs("models", exist_ok=True)

# url and path for model
model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
model_path = "models/yolov8n.pt"


if not os.path.exists(model_path):
    print("Downloading YOLOv8 model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete: yolov8n.pt saved in models/")
else:
    print("YOLOv8 already exists.")
