import torch

if torch.cuda.is_available():
    print(f"✅ CUDA Enabled | GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA Not Available")
    exit()

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


vids = [
    # GOODS
    "C:/Users/Siegfred/programs/bantay-bayan/vids/vecteezy_people-in-waiting-area-of-terminal-d-in-sheremetyevo_28709722.mov",
    "C:/Users/Siegfred/programs/bantay-bayan/vids/vecteezy_traffic-of-passengers-with-luggage-in-amsterdam-airport_28828674.mov",
    "C:/Users/Siegfred/programs/bantay-bayan/vids/vecteezy_padova-italy-18-july-2020-people-wait-for-the-train-bench-in_41477169.mp4",
    "C:/Users/Siegfred/programs/bantay-bayan/vids/vecteezy_venice-italy-6-january-2023-railway-station-interior_41476584.mov",

    # LEAVING
    "C:/Users/Siegfred/programs/bantay-bayan/vids/test-vid.mkv",
    "C:/Users/Siegfred/programs/bantay-bayan/vids/test-vid-2.mkv",
    "C:/Users/Siegfred/programs/bantay-bayan/vids/test-vid-3.mkv",
    
]

# Constants
STATUS_COLORS = {
    "owned": (255, 255, 0),  # Cyan
    "safe": (0, 255, 0),     # Green
    "warning": (0, 255, 255),# Yellow
    "abandoned": (0, 0, 255) # Red
}
CLASS_NAMES = {0: "Human", 24: "Suitcase", 26: "Handbag", 28: "Backpack"}
THRESHOLDS = {
    "stationary": 15,          # Pixels movement
    "owner_distance_ratio": 0.5,  # % of person height
    "abandonment": {           # Seconds
        "green_max": 2,
        "yellow_max": 10,
        "red": 10
    }
}

# Initialize Models
model = YOLO("yolov8l.pt")
person_tracker = DeepSort(max_age=30)
bag_tracker = DeepSort(max_age=30)

# Utility Functions
def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) // 2), int((y1 + y2) // 2))

def get_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) // 2), int(y2))

def calculate_person_size(bbox):
    x1, y1, x2, y2 = bbox
    return (y2 - y1, x2 - x1)

def is_inside(person_bbox, bag_centroid):
    x1, y1, x2, y2 = person_bbox
    bx, by = bag_centroid
    return x1 <= bx <= x2 and y1 <= by <= y2

# Track Storage
tracks = {"people": {}, "bags": {}}

# Main Processing
cap = cv2.VideoCapture(vids[1])
cv2.namedWindow("Smart Baggage Monitoring", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Baggage Monitoring", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
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

    # Tracking
    person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    bag_tracks = bag_tracker.update_tracks(bag_detections, frame=frame)
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Process Bags First
    for track in bag_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        bag_centroid = get_centroid(ltrb)

        # Initialize Bag Track
        if track_id not in tracks["bags"]:
            tracks["bags"][track_id] = {
                "class_id": next((det[2] for det in bag_detections 
                                if get_centroid([det[0][0], det[0][1], 
                                               det[0][0]+det[0][2], 
                                               det[0][1]+det[0][3]]) == bag_centroid), 24),
                "owner": None,
                "last_owner_seen": None,
                "stationary_since": current_time,
                "last_position": bag_centroid,
                "last_seen": current_time
            }

        bag_data = tracks["bags"][track_id]
        class_name = CLASS_NAMES.get(bag_data["class_id"], "Bag")

        # Owner Association
        best_owner = None
        min_distance = float('inf')
        
        for p_id, p_data in tracks["people"].items():
            person_bbox = p_data["bbox"]
            p_height, _ = calculate_person_size(person_bbox)
            
            bottom_center = get_bottom_center(person_bbox)
            person_centroid = get_centroid(person_bbox)
            
            distance_to_feet = np.linalg.norm(np.array(bag_centroid) - np.array(bottom_center))
            distance_to_centroid = np.linalg.norm(np.array(bag_centroid) - np.array(person_centroid))
            current_distance = min(distance_to_feet, distance_to_centroid)
            
            if (current_distance < p_height * THRESHOLDS["owner_distance_ratio"] or 
                is_inside(person_bbox, bag_centroid)) and current_distance < min_distance:
                min_distance = current_distance
                best_owner = p_id

        # Update Ownership
        if best_owner is not None:
            if bag_data["owner"] and bag_data["owner"] in tracks["people"]:
                tracks["people"][bag_data["owner"]]["owned_bags"].discard(track_id)
            bag_data["owner"] = best_owner
            tracks["people"][best_owner]["owned_bags"].add(track_id)
            bag_data["last_owner_seen"] = None
        else:
            if bag_data["owner"]:
                tracks["people"][bag_data["owner"]]["owned_bags"].discard(track_id)
            bag_data["owner"] = None
            if bag_data["last_owner_seen"] is None:
                bag_data["last_owner_seen"] = current_time

        # Abandonment Status System
        abandonment_time = 0
        color = STATUS_COLORS["safe"]
        text = ""
        progress = 0
        
        if bag_data["owner"] is not None:
            color = STATUS_COLORS["owned"]
        elif bag_data["last_owner_seen"]:
            abandonment_time = current_time - bag_data["last_owner_seen"]
            movement = np.linalg.norm(np.array(bag_centroid) - np.array(bag_data["last_position"]))
            
            if movement < THRESHOLDS["stationary"]:
                if abandonment_time > THRESHOLDS["abandonment"]["red"]:
                    color = STATUS_COLORS["abandoned"]
                    text = f"ABANDONED! {abandonment_time:.1f}s"
                    progress = 1.0
                elif abandonment_time > THRESHOLDS["abandonment"]["green_max"]:
                    color = STATUS_COLORS["warning"]
                    text = f"WARNING! {abandonment_time:.1f}s"
                    progress = (abandonment_time - THRESHOLDS["abandonment"]["green_max"]) / \
                              (THRESHOLDS["abandonment"]["red"] - THRESHOLDS["abandonment"]["green_max"])
                else:
                    color = STATUS_COLORS["safe"]
                    progress = abandonment_time / THRESHOLDS["abandonment"]["green_max"]
            else:
                bag_data["last_owner_seen"] = None
                bag_data["last_position"] = bag_centroid
                progress = 0

        # Visualization
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name}#{track_id}", (x1, y1-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Progress bar
        if progress > 0:
            cv2.rectangle(frame, 
                        (x1, y2 + 5), 
                        (int(x1 + (x2-x1) * progress), y2 + 10),
                        color, -1)
        
        if text:
            cv2.putText(frame, text, (x1, y1-40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        if bag_data["owner"] and bag_data["owner"] in tracks["people"]:
            owner_bbox = tracks["people"][bag_data["owner"]]["bbox"]
            ox1, oy1, ox2, oy2 = map(int, owner_bbox)
            owner_centroid = get_centroid(owner_bbox)
            cv2.line(frame, bag_centroid, owner_centroid, STATUS_COLORS["owned"], 2)

        # Update tracking data
        bag_data["last_position"] = bag_centroid
        bag_data["last_seen"] = current_time

    # Process People
    for track in person_tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if track_id not in tracks["people"]:
            tracks["people"][track_id] = {
                "bbox": ltrb,
                "last_seen": current_time,
                "owned_bags": set()
            }
        else:
            tracks["people"][track_id].update({
                "bbox": ltrb,
                "last_seen": current_time
            })
        
        person_color = STATUS_COLORS["owned"] if tracks["people"][track_id]["owned_bags"] else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
        cv2.putText(frame, f"Human#{track_id}", (x1, y1-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)

    # Cleanup old tracks
    current_bag_ids = {t.track_id for t in bag_tracks if t.is_confirmed()}
    for person_id in list(tracks["people"].keys()):
        valid_owned = {bid for bid in tracks["people"][person_id]["owned_bags"] if bid in current_bag_ids}
        tracks["people"][person_id]["owned_bags"] = valid_owned

    for track_type in ["people", "bags"]:
        for t_id in list(tracks[track_type].keys()):
            if current_time - tracks[track_type][t_id].get("last_seen", current_time) > 30:
                del tracks[track_type][t_id]

    # Display
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    cv2.imshow("Smart Baggage Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()