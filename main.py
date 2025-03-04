import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize DeepSORT trackers
person_tracker = DeepSort(max_age=30)
bag_tracker = DeepSort(max_age=30)

# COCO class IDs and names
CLASS_NAMES = {
    0: "Human",
    24: "Suitcase",
    26: "Handbag",
    28: "Backpack"
}

# Track history storage
tracks = {
    "people": {},
    "bags": {}
}

# Thresholds
OWNER_DISTANCE_THRESHOLD = 250  # pixels
ABANDONMENT_TIME_THRESHOLD = 2  # seconds after owner leaves
STATIONARY_THRESHOLD = 15  # pixels movement

def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) // 2), int((y1 + y2) // 2))


vids = [
    # GOODS
    "E:/obs-recording/vecteezy_people-in-waiting-area-of-terminal-d-in-sheremetyevo_28709722.mov",
    "E:/obs-recording/vecteezy_traffic-of-passengers-with-luggage-in-amsterdam-airport_28828674.mov",
    "E:/obs-recording/vecteezy_padova-italy-18-july-2020-people-wait-for-the-train-bench-in_41477169.mp4",
    "E:/obs-recording/vecteezy_venice-italy-6-january-2023-railway-station-interior_41476584.mov",

    # LEAVING
    "E:/obs-recording/test-vid.mkv",
    "E:/obs-recording/test-vid-2.mkv",
    "E:/obs-recording/test-vid-3.mkv",
    
]
cap = cv2.VideoCapture(vids[5])

cv2.namedWindow("Smart Baggage Monitoring", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Baggage Monitoring", 1280, 720)  # Initial size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people and bags
    results = model.predict(frame, classes=[0, 24, 26, 28], conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # Separate detections
    person_detections = []
    bag_detections = []
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confidences, class_ids):
        if cls_id == 0:
            person_detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
        else:
            bag_detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

    # Update trackers
    person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    bag_tracks = bag_tracker.update_tracks(bag_detections, frame=frame)
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Update people tracks AND DRAW THEM
    for track in person_tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Store in tracks dict
        tracks["people"][track_id] = {
            "bbox": ltrb,
            "last_seen": current_time
        }

        # Draw person bounding box (ADDED THIS SECTION)
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
        cv2.putText(frame, f"Human#{track_id}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Process bags
    for track in bag_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        centroid = get_centroid(ltrb)

        # Initialize bag track
        if track_id not in tracks["bags"]:
            tracks["bags"][track_id] = {
                "class_id": next((det[2] for det in bag_detections 
                                if get_centroid(det[0]) == centroid), 24),
                "owner": None,
                "last_owner_seen": None,
                "stationary_since": current_time,
                "last_position": centroid
            }

        bag_data = tracks["bags"][track_id]
        class_name = CLASS_NAMES.get(bag_data["class_id"], "Bag")

        # Update owner status (NEW LOGIC)
        current_owner_valid = False
        if bag_data["owner"] and bag_data["owner"] in tracks["people"]:
            owner_centroid = get_centroid(tracks["people"][bag_data["owner"]]["bbox"])
            distance = np.linalg.norm(np.array(centroid) - np.array(owner_centroid))
            
            if distance <= OWNER_DISTANCE_THRESHOLD:
                current_owner_valid = True
                bag_data["last_owner_seen"] = None
            else:
                # Owner is too far
                current_owner_valid = False
        else:
            current_owner_valid = False

        if not current_owner_valid:
            # Find new owner
            min_distance = float('inf')
            new_owner_id = None
            for p_id, p_data in tracks["people"].items():
                p_centroid = get_centroid(p_data["bbox"])
                distance = np.linalg.norm(np.array(centroid) - np.array(p_centroid))
                
                if distance < min_distance and distance < OWNER_DISTANCE_THRESHOLD:
                    min_distance = distance
                    new_owner_id = p_id
            
            if new_owner_id is not None:
                # Assign new owner
                bag_data["owner"] = new_owner_id
                bag_data["last_owner_seen"] = None
            else:
                # No valid owner
                bag_data["owner"] = None
                if bag_data["last_owner_seen"] is None:
                    bag_data["last_owner_seen"] = current_time

        # Check abandonment conditions
        abandonment_time = 0
        color = (0, 255, 0)  # Green
        text = ""
        
        if bag_data["last_owner_seen"]:
            abandonment_time = current_time - bag_data["last_owner_seen"]
            movement = np.linalg.norm(np.array(centroid) - np.array(bag_data["last_position"]))
            
            if abandonment_time > ABANDONMENT_TIME_THRESHOLD and movement < STATIONARY_THRESHOLD:
                color = (0, 0, 255)  # Red
                text = f"ABANDONED! {abandonment_time:.1f}s"

        # Draw elements
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name}#{track_id}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if bag_data["owner"] and bag_data["owner"] in tracks["people"]:
            # Draw owner connection line
            owner_bbox = tracks["people"][bag_data["owner"]]["bbox"]
            ox1, oy1, ox2, oy2 = map(int, owner_bbox)
            
            # Get integer centroids
            bag_centroid = (int(centroid[0]), int(centroid[1]))  # Already calculated as integers
            owner_centroid = get_centroid(owner_bbox)
            
            cv2.line(frame, bag_centroid, owner_centroid, (255, 255, 0), 2)
            cv2.putText(frame, f"Owner#{bag_data['owner']}", 
                    (ox1, oy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if text:
            cv2.putText(frame, text, (x1, y1-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update position history
        bag_data["last_position"] = centroid

    # Cleanup old tracks
    for track_type in ["people", "bags"]:
        for t_id in list(tracks[track_type].keys()):
            if current_time - tracks[track_type][t_id].get("last_seen", current_time) > 30:
                del tracks[track_type][t_id]


    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Scale down

    cv2.imshow("Smart Baggage Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()