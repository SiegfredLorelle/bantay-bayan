import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Target classes (COCO IDs for bags)
TARGET_CLASSES = [24, 26, 28]

# Track history storage
track_history = {}

# Thresholds
STATIONARY_THRESHOLD = 5  # 5 seconds
MOVEMENT_THRESHOLD = 15    # pixels

# Video input
cap = cv2.VideoCapture("E:/obs-recording/test-vid.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model.predict(frame, classes=TARGET_CLASSES, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # Convert to DeepSORT format
    detections = [([x1, y1, x2-x1, y2-y1], conf, cls_id) 
                for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_ids)]

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Initialize new tracks
        if track_id not in track_history:
            track_history[track_id] = {
                "stationary_time": current_time,
                "last_position": (ltrb[0], ltrb[1])
            }

        # Calculate movement
        last_pos = track_history[track_id]["last_position"]
        current_pos = (ltrb[0], ltrb[1])
        movement = np.sqrt((current_pos[0]-last_pos[0])**2 + (current_pos[1]-last_pos[1])**2)

        # Default: green box for all bags
        color = (0, 255, 0)  # Green
        text = ""

        # Check abandonment
        if movement < MOVEMENT_THRESHOLD:
            stationary_time = current_time - track_history[track_id]["stationary_time"]
            if stationary_time > STATIONARY_THRESHOLD:
                color = (0, 0, 255)  # Red
                text = f"ABANDONED! {stationary_time:.1f}s"
        else:
            track_history[track_id]["stationary_time"] = current_time

        # Always draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add text only for abandoned bags
        if text:
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update position
        track_history[track_id]["last_position"] = current_pos

    # Display
    cv2.imshow("Suspicious Bag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()