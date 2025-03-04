import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT trackers
person_tracker = DeepSort(max_age=30)
bag_tracker = DeepSort(max_age=30)

# COCO class IDs
PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28]  # suitcase, handbag, backpack

# Track history for bags
bag_track_history = {}

# Thresholds
STATIONARY_THRESHOLD = 5  # seconds
MOVEMENT_THRESHOLD = 15   # pixels

# Video input
cap = cv2.VideoCapture("E:/obs-recording/test-vid.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people and bags
    results = model.predict(frame, classes=[PERSON_CLASS] + BAG_CLASSES, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # Separate detections
    person_detections = []
    bag_detections = []
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_ids):
        if cls_id == PERSON_CLASS:
            person_detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
        else:
            bag_detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

    # Update trackers
    person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    bag_tracks = bag_tracker.update_tracks(bag_detections, frame=frame)
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Draw people (blue boxes)
    for track in person_tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue

    # Process bags
    for track in bag_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Initialize new bag tracks
        if track_id not in bag_track_history:
            bag_track_history[track_id] = {
                "stationary_time": current_time,
                "last_position": (ltrb[0], ltrb[1])
            }

        # Calculate movement
        last_pos = bag_track_history[track_id]["last_position"]
        current_pos = (ltrb[0], ltrb[1])
        movement = np.sqrt((current_pos[0]-last_pos[0])**2 + (current_pos[1]-last_pos[1])**2)

        # Default: green box for bags
        color = (0, 255, 0)
        text = ""

        # Check abandonment
        if movement < MOVEMENT_THRESHOLD:
            stationary_time = current_time - bag_track_history[track_id]["stationary_time"]
            if stationary_time > STATIONARY_THRESHOLD:
                color = (0, 0, 255)  # Red
                text = f"ABANDONED! {stationary_time:.1f}s"
        else:
            bag_track_history[track_id]["stationary_time"] = current_time

        # Draw bag box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        bag_track_history[track_id]["last_position"] = current_pos

    # Display
    cv2.imshow("Suspicious Bag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()