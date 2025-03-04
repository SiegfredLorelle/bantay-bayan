import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model (pre-trained on COCO)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Adjust max_age for track persistence

# Define target classes (COCO class IDs for 'suitcase', 'backpack', etc.)
TARGET_CLASSES = [24, 26, 28]  # 24: suitcase, 26: handbag, 28: backpack (COCO IDs)

# Store tracked objects and their positions
track_history = {}  # {track_id: {"positions": [], "stationary_time": 0}}

# Thresholds
STATIONARY_THRESHOLD = 5  # Seconds to consider an object as abandoned
MOVEMENT_THRESHOLD = 15   # Pixels movement to reset stationary timer

# Open video source (file or webcam)
input_path = "E:/obs-recording/test-vid.mkv"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Run YOLOv8 inference
    results = model.predict(frame, classes=TARGET_CLASSES, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # Convert to DeepSORT format
    detections = []
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process tracks
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current timestamp in seconds
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box

        # Initialize new tracks
        if track_id not in track_history:
            track_history[track_id] = {
                "positions": [],
                "stationary_time": current_time,
                "last_position": (ltrb[0], ltrb[1])
            }
            continue

        # Calculate movement from last position
        last_pos = track_history[track_id]["last_position"]
        current_pos = (ltrb[0], ltrb[1])
        movement = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)

        # Update stationary time
        if movement < MOVEMENT_THRESHOLD:
            stationary_time = current_time - track_history[track_id]["stationary_time"]
            if stationary_time > STATIONARY_THRESHOLD:
                # Trigger alert for abandoned bag
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
                cv2.putText(frame, f"ABANDONED! {stationary_time:.1f}s", (int(ltrb[0]), int(ltrb[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            track_history[track_id]["stationary_time"] = current_time  # Reset timer

        # Update last known position
        track_history[track_id]["last_position"] = current_pos

    # Display
    cv2.imshow("Suspicious Bag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop.")
        break

cap.release()
cv2.destroyAllWindows()