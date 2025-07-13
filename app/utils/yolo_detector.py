import torch
from ultralytics import YOLO
import numpy as np
from huggingface_hub import hf_hub_download

yolo_model = hf_hub_download(
    repo_id="balaji2003/yolov8x-model",
    filename="yolov8x.pt"
)

model = YOLO(yolo_model)  

def detect_license_plate(image_path):
    results = model(image_path)
    if results is None or len(results) == 0:
        return None

    boxes = results[0].boxes

    if boxes is None or boxes.data is None or boxes.data.shape[0] == 0:
        return None 
    predictions = boxes.data.cpu().numpy()  
    return predictions
