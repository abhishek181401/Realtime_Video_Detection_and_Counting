import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n.pt"
model = YOLO(MODEL_PATH)

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
tracked_ids = set()

def detect_vehicles(frame):
 
    global tracked_ids, vehicle_counts

    results = model(frame) 
    new_tracked_ids = set() 

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  
            class_name = model.names[class_id] 
            obj_id = int(box.id[0]) if box.id is not None else None  

            if class_name in VEHICLE_CLASSES:
                new_tracked_ids.add(obj_id) 
                
                if obj_id not in tracked_ids and obj_id is not None:
                    vehicle_counts[class_name] += 1 

               
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_ids = new_tracked_ids  
    return frame, vehicle_counts  
