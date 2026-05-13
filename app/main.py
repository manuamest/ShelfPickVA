import cv2
import json
import sys
import numpy as np
import os
from ultralytics import YOLO
from scipy.spatial import distance

# Constants
CONFIG_FILE = 'bottle_config.json'
YOLO_MODEL_PATH = 'models/model.pt' # Placeholder, user should replace with SKU-110k model
YOLO_POSE_PATH = 'models/yolov8n-pose.pt'
CONF_THRESHOLD = 0.4

class BottleDetector:
    def __init__(self, model_path):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        # Run inference with ByteTrack
        # persist=True is important for tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                
                if conf > CONF_THRESHOLD:
                    detections.append({
                        "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                        "conf": float(conf),
                        "class": int(cls),
                        "track_id": track_id
                    })
        return detections

class HandDetector:
    def __init__(self, model_path):
        print(f"Loading YOLO Pose model from {model_path}...")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        hands_list = []
        
        for r in results:
            keypoints = r.keypoints
            if keypoints is None: continue
            
            # Iterate over each detected person/pose
            for kps in keypoints.data:
                # kps is (17, 3) -> x, y, conf
                # Wrist indices: 9 (left), 10 (right)
                # We can also use elbow (7, 8) to estimate direction
                
                for wrist_idx in [9, 10]:
                    if kps[wrist_idx][2] > 0.5: # Confidence check
                        wx, wy = int(kps[wrist_idx][0]), int(kps[wrist_idx][1])
                        
                        # Estimate hand box around wrist
                        # This is a heuristic since pose doesn't give hand box
                        box_size = 60
                        x_min = max(0, wx - box_size)
                        y_min = max(0, wy - box_size)
                        x_max = min(frame.shape[1], wx + box_size)
                        y_max = min(frame.shape[0], wy + box_size)
                        
                        hands_list.append([x_min, y_min, x_max, y_max])
                        
        return hands_list

class InteractionLogic:
    def __init__(self, config):
        self.config = config
        self.rois = config.get("rois", [])
        self.roi_dwell_times = {roi['label']: 0 for roi in self.rois if 'label' in roi}
        # Parse histograms from config
        for roi in self.rois:
            if "histogram" in roi:
                # Convert list back to numpy array
                hist = np.array(roi["histogram"], dtype=np.float32)
                roi["hist_np"] = hist.reshape((180, 256))
            else:
                roi["hist_np"] = None
            
            # Load reference image implicitly based on label
            roi["ref_img"] = None
            import re
            safe_label = re.sub(r'[^a-zA-Z0-9]', '_', roi['label'])
            image_filename = f"roi_bottle_{safe_label}.jpg"
            
            # 1. Try in roi_images/ relative to CWD
            local_path = os.path.join("roi_images", image_filename)
            if os.path.exists(local_path):
                roi["ref_img"] = cv2.imread(local_path)
                # print(f"Loaded ROI image: {local_path}")
            else:
                # 2. Try in ../roi_images/ relative to CWD
                parent_path = os.path.join("..", "roi_images", image_filename)
                if os.path.exists(parent_path):
                    roi["ref_img"] = cv2.imread(parent_path)
                    # print(f"Loaded ROI image: {parent_path}")
                else:
                    print(f"Warning: ROI image not found for label '{roi['label']}': {local_path}")

    def check_hand_in_roi(self, hand_box):
        # Check if hand center is inside any defined ROI
        hx1, hy1, hx2, hy2 = hand_box
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2
        
        for roi in self.rois:
            if 'points' in roi:
                # Polygon ROI
                pts = np.array(roi['points'], dtype=np.int32)
                # measureDist=False returns +1 if inside, -1 if outside, 0 if on edge
                result = cv2.pointPolygonTest(pts, (h_cx, h_cy), False)
                if result >= 0:
                    return roi['label']
            elif 'rect' in roi:
                # Legacy Rect ROI support
                rx, ry, rw, rh = roi['rect']
                if rx <= h_cx <= rx + rw and ry <= h_cy <= ry + rh:
                    return roi['label']
                if rx <= h_cx <= rx + rw and ry <= h_cy <= ry + rh:
                    return roi['label']
        return None

    def update_dwell_times(self, active_label):
        # Increment active label, decay others
        for label in self.roi_dwell_times:
            if label == active_label:
                self.roi_dwell_times[label] = min(100, self.roi_dwell_times[label] + 2) # Cap at 100
            else:
                self.roi_dwell_times[label] = max(0, self.roi_dwell_times[label] - 1) # Decay

    def find_closest_bottle(self, hand_box, bottle_detections):
        # Find bottle detection closest to hand center
        if not bottle_detections:
            return None
            
        hx1, hy1, hx2, hy2 = hand_box
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2
        
        min_dist = float('inf')
        closest_bottle = None
        
        for bottle in bottle_detections:
            bx1, by1, bx2, by2 = bottle['bbox']
            b_cx = (bx1 + bx2) // 2
            b_cy = (by1 + by2) // 2
            
            dist = np.sqrt((h_cx - b_cx)**2 + (h_cy - b_cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_bottle = bottle
                
        return closest_bottle

    def get_similarity_scores(self, bottle_img):
        if bottle_img.size == 0:
            return []
            
        hsv_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        
        scores = []
        for roi in self.rois:
            if roi.get("hist_np") is not None:
                base_score = cv2.compareHist(hist, roi["hist_np"], cv2.HISTCMP_CORREL)
                
                # Add dwell time bonus
                # Max bonus of 0.5 if dwell time is 100
                dwell_bonus = (self.roi_dwell_times.get(roi["label"], 0) / 100.0) * 0.5
                final_score = base_score + dwell_bonus
                
                scores.append({
                    "label": roi["label"],
                    "score": final_score,
                    "ref_img": roi.get("ref_img")
                })
        
        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:5]

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (200, 200, 200)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r) # OpenCV uses BGR

def draw_bottom_panel(frame, scores, panel_height=300):
    h, w, c = frame.shape
    panel = np.zeros((panel_height, w, c), dtype=np.uint8)
    panel[:] = (30, 30, 30) # Dark gray background
    
    # Calculate cell dimensions
    num_cells = 5
    cell_width = w // num_cells
    cell_height = panel_height
    
    # Draw cells
    for i in range(num_cells):
        x_start = i * cell_width
        x_end = (i + 1) * cell_width
        
        # Draw separator
        if i > 0:
            cv2.line(panel, (x_start, 0), (x_start, panel_height), (100, 100, 100), 1)
            
        if i < len(scores):
            item = scores[i]
            
            # Draw Reference Image (Centered)
            img_size = min(cell_width, cell_height) - 60
            if item["ref_img"] is not None:
                ref_h, ref_w = item["ref_img"].shape[:2]
                scale = min(img_size/ref_h, img_size/ref_w)
                new_w, new_h = int(ref_w * scale), int(ref_h * scale)
                resized_ref = cv2.resize(item["ref_img"], (new_w, new_h))
                
                y_offset = (cell_height - new_h) // 2 - 20
                x_offset = x_start + (cell_width - new_w) // 2
                panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_ref
            else:
                # Placeholder
                cv2.rectangle(panel, (x_start + (cell_width-img_size)//2, (cell_height-img_size)//2 - 20), 
                              (x_start + (cell_width+img_size)//2, (cell_height+img_size)//2 - 20 + img_size), (50, 50, 50), -1)
                cv2.putText(panel, "No Img", (x_start + cell_width//2 - 40, cell_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

            # Label
            label = item["label"]
            if len(label) > 15: label = label[:12] + "..."
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x_start + (cell_width - text_size[0]) // 2
            cv2.putText(panel, label, (text_x, cell_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Score Bar
            score = max(0, min(1.0, item["score"])) # Clamp 0-1
            bar_w = int((cell_width - 40) * score)
            bar_start_x = x_start + 20
            cv2.rectangle(panel, (bar_start_x, cell_height - 40), (bar_start_x + cell_width - 40, cell_height - 25), (50, 50, 50), -1)
            cv2.rectangle(panel, (bar_start_x, cell_height - 40), (bar_start_x + bar_w, cell_height - 25), (0, 255, 0), -1)
            
            # Score Text
            score_text = f"{score:.2f}"
            cv2.putText(panel, score_text, (bar_start_x + cell_width - 40 - 50, cell_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return np.vstack((frame, panel))

def draw_transparent_box(frame, box, color, alpha=0.5):
    x1, y1, x2, y2 = box
    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    # Blend
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    # Load Config
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_FILE} not found. Run config_tool.py first.")
        sys.exit(1)

    # Initialize Detectors
    bottle_detector = BottleDetector(YOLO_MODEL_PATH)
    hand_detector = HandDetector(YOLO_POSE_PATH)
    logic = InteractionLogic(config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        sys.exit(1)
        
    # Output saver
    width = int(cap.get(3))
    height = int(cap.get(4))
    panel_height = 400
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height + panel_height))

    # Resizable Window
    cv2.namedWindow("Bottle Detection System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bottle Detection System", 1280, 900)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect Bottles
        bottles = bottle_detector.detect(frame)
        
        # 2. Detect Hands
        hands = hand_detector.detect(frame)
        
        # 3. Logic & Visualization
        current_scores = []
        
        # Draw defined ROIs
        for roi in config.get("rois", []):
            color = hex_to_bgr(roi.get("color", "#C8C8C8"))
            if 'points' in roi:
                pts = np.array(roi['points'], dtype=np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, roi['label'], (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif 'rect' in roi:
                rx, ry, rw, rh = roi['rect']
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 2)
                cv2.putText(frame, roi['label'], (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Bottle Detections
        for b in bottles:
            bx1, by1, bx2, by2 = b['bbox']
            track_id = b.get('track_id', -1)
            # Transparent Gray Box
            draw_transparent_box(frame, (bx1, by1, bx2, by2), (128, 128, 128), alpha=0.4)
            # Thin outline
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
            cv2.putText(frame, f"ID:{track_id}", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        # Process Hands
        for hand_box in hands:
            hx1, hy1, hx2, hy2 = hand_box
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
            
            # Check if hand is in a specific ROI (Picking action?)
            roi_label = logic.check_hand_in_roi(hand_box)
            logic.update_dwell_times(roi_label) # Update dwell times
            
            if roi_label:
                cv2.putText(frame, f"Hand in: {roi_label}", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Find closest bottle to hand to identify it
            closest_bottle = logic.find_closest_bottle(hand_box, bottles)
            if closest_bottle:
                cbx1, cby1, cbx2, cby2 = closest_bottle['bbox']
                # Draw line from hand to bottle
                h_cx, h_cy = (hx1+hx2)//2, (hy1+hy2)//2
                b_cx, b_cy = (cbx1+cbx2)//2, (cby1+cby2)//2
                cv2.line(frame, (h_cx, h_cy), (b_cx, b_cy), (255, 0, 255), 2)
                
                # Extract bottle crop for similarity check
                bottle_crop = frame[cby1:cby2, cbx1:cbx2]
                current_scores = logic.get_similarity_scores(bottle_crop)
                
                # Show top match on bottle
                if current_scores:
                    best_match = current_scores[0]
                    if best_match["score"] > 0.5:
                        cv2.putText(frame, f"Sim: {best_match['label']}", (b_cx, b_cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw Bottom Panel
        final_frame = draw_bottom_panel(frame, current_scores, panel_height)

        cv2.imshow("Bottle Detection System", final_frame)
        out.write(final_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
