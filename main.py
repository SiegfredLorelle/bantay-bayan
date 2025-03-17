import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Global flag for stopping monitoring
stop_monitoring = False

def run_monitoring(video_path, thresholds):
    global stop_monitoring
    stop_monitoring = False
    
    # Initialize Models
    model = YOLO("yolov8l.pt")
    person_tracker = DeepSort(max_age=30)
    bag_tracker = DeepSort(max_age=30)

    # Constants
    CLASS_NAMES = {
        0: "Person",
        24: "Backpack",
        26: "Handbag",
        28: "Suitcase"
    }
    
    STATUS_COLORS = {
        "safe": (0, 255, 0),      # Green
        "warning": (0, 165, 255),  # Orange
        "abandoned": (0, 0, 255),  # Red
        "owned": (255, 255, 0)     # Cyan
    }

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
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("BantayBayan Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BantayBayan Monitoring", 1280, 720)

    while cap.isOpened() and not stop_monitoring:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        results = model.predict(frame, classes=[0, 24, 26, 28], conf=thresholds["detection_confidence"])
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
                
                if (current_distance < p_height * thresholds["owner_distance_ratio"] or 
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
                if bag_data["owner"] and bag_data["owner"] in tracks["people"]:
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
                
                if movement < thresholds["stationary"]:
                    if abandonment_time > thresholds["abandonment"]["red"]:
                        color = STATUS_COLORS["abandoned"]
                        text = f"ABANDONED! {abandonment_time:.1f}s"
                        progress = 1.0
                    elif abandonment_time > thresholds["abandonment"]["green_max"]:
                        color = STATUS_COLORS["warning"]
                        text = f"WARNING! {abandonment_time:.1f}s"
                        progress = (abandonment_time - thresholds["abandonment"]["green_max"]) / \
                                  (thresholds["abandonment"]["red"] - thresholds["abandonment"]["green_max"])
                    else:
                        color = STATUS_COLORS["safe"]
                        progress = abandonment_time / thresholds["abandonment"]["green_max"]
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
        cv2.imshow("BantayBayan Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to stop monitoring
def stop_monitoring_process():
    global stop_monitoring
    stop_monitoring = True

# Streamlit UI
def main():
    st.set_page_config(page_title="BantayBayan Monitoring", layout="wide")
    
    st.title("BantayBayan Monitoring System")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Video source selection
        video_source = st.radio("Select Video Source", ["Upload Video", "Use Webcam", "Use Video Path"])
        
        video_path = None
        
        if video_source == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "mkv"])
            if uploaded_file:
                # Save uploaded file to temp location
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                st.success(f"Video uploaded successfully!")
        elif video_source == "Use Webcam":
            webcam_id = st.number_input("Webcam ID", min_value=0, value=0, step=1)
            video_path = int(webcam_id)
        else:  # Use Video Path
            video_path = st.text_input("Enter video file path", "C:/path/to/your/video.mp4")
            if not os.path.exists(video_path) and video_path != "C:/path/to/your/video.mp4":
                st.error("Video file not found!")
        
        # Detection thresholds
        st.subheader("Detection Settings")
        detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
        
        # Tracking thresholds
        st.subheader("Tracking Settings")
        owner_distance_ratio = st.slider("Owner Distance Ratio", 0.1, 3.0, 0.5, 0.1, 
                                        help="How close a person needs to be to a bag to be considered its owner")
        stationary_threshold = st.slider("Stationary Threshold (pixels)", 1, 20, 5, 1,
                                        help="Maximum movement (in pixels) for a bag to be considered stationary")
        
        # Abandonment thresholds
        st.subheader("Abandonment Settings")
        green_max = st.slider("Warning Time (seconds)", 2.0, 60.0, 3.0, 1.0,
                             help="Time after which a stationary bag without owner will trigger a warning")
        red_threshold = st.slider("Abandoned Time (seconds)", 10.0, 120.0, 10.0, 1.0,
                                 help="Time after which a stationary bag without owner will be considered abandoned")
        
        thresholds = {
            "detection_confidence": detection_confidence,
            "owner_distance_ratio": owner_distance_ratio,
            "stationary": stationary_threshold,
            "abandonment": {
                "green_max": green_max,
                "red": red_threshold
            }
        }
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Monitor Control")
        
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
            
        # Check if we can start monitoring
        can_start = False
        if video_source == "Upload Video" and uploaded_file:
            can_start = True
        elif video_source == "Use Webcam":
            can_start = True
        elif video_source == "Use Video Path" and os.path.exists(video_path) and video_path != "path/to/your/video.mp4":
            can_start = True
            
        # Start/Stop buttons
        if not can_start:
            st.warning("Please provide a valid video source to start monitoring")
            st.button("Start Monitoring", disabled=True)
        else:
            if not st.session_state.monitoring_active:
                if st.button("Start Monitoring"):
                    st.session_state.monitoring_active = True
                    # Start monitoring in a separate thread
                    monitoring_thread = threading.Thread(
                        target=run_monitoring, 
                        args=(video_path, thresholds)
                    )
                    monitoring_thread.start()
                    st.success("Monitoring started! OpenCV window should appear.")
            else:
                if st.button("Stop Monitoring"):
                    st.session_state.monitoring_active = False
                    stop_monitoring_process()
                    st.info("Monitoring stopped")
    
    with col2:
        st.header("Status")
        status_placeholder = st.empty()
        
        if st.session_state.monitoring_active:
            status_placeholder.success("Monitoring Active")
        else:
            status_placeholder.info("Monitoring Inactive")
            
        # Display current settings
        st.subheader("Current Settings")
        st.json(thresholds)
    
    # Information section
    st.header("System Information")
    with st.expander("About BantayBayan Monitoring"):
        st.markdown("""
        ### BantayBayan Monitoring System
        
        This system tracks people and their baggage in public spaces, detecting potential abandoned items.
        
        #### Features:
        - Real-time detection of people and baggage items (backpacks, handbags, suitcases)
        - Ownership tracking - determining which person owns which bag
        - Abandoned baggage alerts with configurable thresholds
        - Visual indicators for bag status
        
        #### Status Colors:
        - 🟢 **Green**: Bag is safe or has an owner nearby
        - 🟠 **Orange**: Warning, bag may be abandoned
        - 🔴 **Red**: Alert, bag is considered abandoned
        - 🔵 **Cyan**: Bag is currently associated with an owner
        
        #### Tips:
        - Adjust thresholds based on the environment and camera position
        - Higher detection confidence may miss some objects but gives fewer false positives
        - The owner distance ratio determines how close a person must be to be considered a bag's owner
        """)

if __name__ == "__main__":
    main()