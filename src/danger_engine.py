from ultralytics import YOLO
import torch
# FIXED: Removed '.' before config
from config import YOLO_MODEL_PATH, DANGER_CLASSES, CONFIDENCE_THRESHOLD, USE_GPU

class DangerEngine:
    def __init__(self):
        print("[System] Initializing Danger Engine (YOLO)...")
        
        # Force download if missing
        self.model = YOLO('yolov8m.pt') 
        
        # GPU Acceleration Logic
        if USE_GPU and torch.cuda.is_available():
            print(f"[System] ✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
            self.model.to('cuda')
        else:
            print("[System] ⚠️ GPU not found or disabled. Using CPU.")

    def analyze(self, frame):
        """
        Returns:
            - danger_detected (bool)
            - closest_object (dict or None)
            - detections (list of raw boxes)
        """
        # Run inference
        # stream=True is faster, agnostic=True reduces flickering
        results = self.model(frame, verbose=False, stream=True, agnostic_nms=True)
        
        height, width = frame.shape[:2]
        center_zone_start = width * 0.25
        center_zone_end = width * 0.75

        danger_detected = False
        danger_label = ""
        closest_obj = None
        max_area = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < CONFIDENCE_THRESHOLD: continue

                # Bounding Box Coords
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) // 2
                
                # 1. Danger Check
                if cls_id in DANGER_CLASSES:
                    # Is it in front of us?
                    if center_zone_start < center_x < center_zone_end:
                        danger_detected = True
                        danger_label = self.model.names[cls_id]

                # 2. Sonar Tracking (Largest Object)
                if area > max_area:
                    max_area = area
                    closest_obj = {
                        "center_x": center_x,
                        "area": area,
                        "box": (x1, y1, x2, y2),
                        "label": self.model.names[cls_id]
                    }

        return danger_detected, danger_label, closest_obj