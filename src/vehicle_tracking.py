import cv2
import numpy as np
from ultralytics import YOLO
from supervision.detection.core import Detections
from supervision.tracker.core import Tracker

class VehicleTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  
        self.tracker = Tracker()  
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        self.tracked_vehicles = {}

    def track_vehicles(self, frame):
        results = self.model(frame) 
        detections = Detections.from_ultralytics(results)  

        tracked_objects = self.tracker.update_with_detections(detections) 
        vehicle_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

        for track in tracked_objects:
            x1, y1, x2, y2, track_id, cls_id, conf = track  
            cls_id = int(cls_id)

            if cls_id in self.vehicle_classes:
                label = self.vehicle_classes[cls_id]
                vehicle_counts[label] += 1

                if track_id not in self.tracked_vehicles:
                    self.tracked_vehicles[track_id] = label

               
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, vehicle_counts
