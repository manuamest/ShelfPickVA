import cv2
import json
import sys
import numpy as np
import os

try:
    from detectors import HandDetector, BottleSegmenter
    from logic import InteractionLogic
except ImportError:
    # Fallback for when running from root as module or different env
    from app.detectors import HandDetector, BottleSegmenter
    from app.logic import InteractionLogic

# Constants
CONFIG_FILE = 'cfgs/bottle_config.json'
YOLO_MODEL_PATH = 'models/model.pt' # Placeholder, user should replace with SKU-110k model
YOLO_POSE_PATH = 'models/yolov8n-pose.pt'

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (200, 200, 200)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r) # OpenCV uses BGR

def draw_bottom_panel(frame, scores_dict, panel_height=400):
    h, w, c = frame.shape
    panel = np.zeros((panel_height, w, c), dtype=np.uint8)
    panel[:] = (30, 30, 30) # Dark gray background
    
    # Check if we have data
    if not scores_dict:
        return np.vstack((frame, panel))
        
    visual_scores = scores_dict.get("visual_scores", [])
    final_scores = scores_dict.get("final_scores", [])
    
    # 2 Rows: Top = Visual Sim, Bottom = Final Prob
    row_height = panel_height // 2
    
    # --- Row 1: Visual Similarity ---
    cv2.putText(panel, "Visual Similarity Ranking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    _draw_score_row(panel, visual_scores, 0, row_height, w)
    
    # --- Row 2: Final Probability ---
    cv2.putText(panel, "Final Probability (Logic Adjusted)", (10, row_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    _draw_score_row(panel, final_scores, row_height, row_height, w)

    return np.vstack((frame, panel))

def _draw_score_row(panel, scores, y_start, row_height, w):
    num_cells = 5
    cell_width = w // num_cells
    
    for i in range(num_cells):
        x_start = i * cell_width
        
        # draw separator
        cv2.line(panel, (x_start, y_start), (x_start, y_start + row_height), (100, 100, 100), 1)
        
        if i < len(scores):
            item = scores[i]
            # Content area
            cy_start = y_start + 40
            c_height = row_height - 40
            
            # Draw Image
            img_size = min(cell_width, c_height) - 40
            if item["ref_img"] is not None:
                ref_h, ref_w = item["ref_img"].shape[:2]
                scale = min(img_size/ref_h, img_size/ref_w)
                new_w, new_h = int(ref_w * scale), int(ref_h * scale)
                resized_ref = cv2.resize(item["ref_img"], (new_w, new_h))
                
                y_offset = cy_start + (c_height - new_h) // 2 - 10
                x_offset = x_start + (cell_width - new_w) // 2
                panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_ref
            
            # Label
            label = item["label"]
            if len(label) > 15: label = label[:12] + "..."
            cv2.putText(panel, label, (x_start + 10, y_start + row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Score
            score_txt = f"{item['score']:.2f}"
            cv2.putText(panel, score_txt, (x_start + cell_width - 60, y_start + row_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


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
    
    collect_data = False
    if "--collect-data" in sys.argv:
        collect_data = True
        print("Data Collection Mode: Enabled. Saving hand crops to data/hand_crops_raw/")
        os.makedirs("data/hand_crops_raw", exist_ok=True)

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_FILE} not found. Run config_tool.py first.")
        sys.exit(1)

    # Initialize Detectors
    # bottle_detector = BottleDetector(YOLO_MODEL_PATH)
    bottle_segmenter = BottleSegmenter() # Auto downloads yolov8n-seg.pt
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
            
        clean_frame = frame.copy()
            
        # 1. Detect Hands
        hands = hand_detector.detect(frame)
        
        # 2. Logic & Visualization
        active_rois = logic.check_roi_occupancy(hands)
        current_scores_dict = {}
        
        # Draw defined ROIs
        for roi in config.get("rois", []):
            color = hex_to_bgr(roi.get("color", "#C8C8C8"))
            
            # Dim color if not active
            if roi['label'] not in active_rois:
                # Make it darker/grayer
                color = tuple([c//2 for c in color])
                
            if 'points' in roi:
                pts = np.array(roi['points'], dtype=np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
                cv2.putText(frame, roi['label'], (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif 'rect' in roi:
                rx, ry, rw, rh = roi['rect']
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 2)
                cv2.putText(frame, roi['label'], (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Process Hands & Segmentation
        for hand_box in hands:
            # hand_box has 5 elements: x1, y1, x2, y2, track_id
            hx1, hy1, hx2, hy2, track_id = hand_box
            
            color = (0, 0, 255)
            if track_id != -1:
                # Generate color based on track_id
                np.random.seed(track_id)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (hx1, hy1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check if hand is in a specific ROI (Picking action?)
            roi_label = logic.check_hand_in_roi(hand_box)
            logic.update_state(roi_label) # Update dwell times
            
            if roi_label:
                cv2.putText(frame, f"Hand in: {roi_label}", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            # --- Hand Crop & Segmentation Logic ---
            # Calculate Hand Center
            h_cx, h_cy = (hx1+hx2)//2, (hy1+hy2)//2
            h_w = hx2 - hx1
            h_h = hy2 - hy1
            
            # Custom directional margins
            # Legacy was 6x total size centered -> ~2.5x margin on each side
            # New req: North=Equal(2.5x), South=Remove(0), East/West=Half(1.25x)
            
            ref_size = max(h_w, h_h)
            
            margin_north = int(ref_size * 2.5)
            margin_south = 0
            margin_side = int(ref_size * 1.25)
            
            x_min = max(0, hx1 - margin_side)
            y_min = max(0, hy1 - margin_north)
            x_max = min(frame.shape[1], hx2 + margin_side)
            y_max = min(frame.shape[0], hy2 + margin_south)
            
            if x_max > x_min and y_max > y_min:
                hand_crop = clean_frame[y_min:y_max, x_min:x_max]
                
                # Data Collection
                if collect_data:
                    # Save crop
                    import time
                    timestamp = int(time.time() * 1000)
                    filename = f"data/hand_crops_raw/crop_{timestamp}_{track_id}.jpg"
                    cv2.imwrite(filename, hand_crop)

                # Run Segmentation
                masked_bottle, mask = bottle_segmenter.segment_bottle(hand_crop)
                
                # Visual Debug: Draw crop box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                
                if masked_bottle is not None:
                    # Found a bottle in hand!
                    current_scores_dict = logic.get_annotated_scores(masked_bottle, active_rois)
                    
                    # Overlay mask on main frame for visualization
                    # Mask is black/white, we want colored overlay
                    # Create colored mask overlay (Green)
                    color_mask = np.zeros_like(hand_crop)
                    color_mask[:, :] = (0, 255, 0) # BGR
                    
                    # Apply mask
                    mask_bool = mask > 0
                    
                    # Blend into frame
                    roi_slice = frame[y_min:y_max, x_min:x_max]
                    roi_slice[mask_bool] = cv2.addWeighted(roi_slice[mask_bool], 0.7, color_mask[mask_bool], 0.3, 0)
                    
                    # Show top match near hand
                    if current_scores_dict.get("final_scores"):
                        best_match = current_scores_dict["final_scores"][0]
                        cv2.putText(frame, f"Holding: {best_match['label']}", (x_max, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {best_match['score']:.2f}", (x_max, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Draw Bottom Panel
        final_frame = draw_bottom_panel(frame, current_scores_dict, panel_height)

        cv2.imshow("Bottle Detection System", final_frame)
        out.write(final_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
