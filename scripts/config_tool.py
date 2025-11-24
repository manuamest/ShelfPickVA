import cv2
import json
import os
import sys

CONFIG_FILE = 'bottle_config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"rois": []}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {CONFIG_FILE}")

def select_rois(frame):
    print("Select ROIs. Draw a rectangle and press SPACE or ENTER. Press 'c' to cancel selection.")
    rois = cv2.selectROIs("Select ROIs", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROIs")
    return rois

def get_label_for_roi(roi, frame):
    x, y, w, h = roi
    roi_img = frame[y:y+h, x:x+w]
    cv2.imshow("Selected ROI", roi_img)
    print("Enter label for this ROI in the terminal:")
    cv2.waitKey(100) # Give time for window to render
    label = input(f"Label for ROI at {roi}: ")
    cv2.destroyWindow("Selected ROI")
    return label

def main():
    if len(sys.argv) < 2:
        print("Usage: python config_tool.py <video_path_or_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    
    # Check if input is image or video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video/image file: {input_path}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Could not read the first frame.")
        sys.exit(1)
    
    # Resize for easier selection if too big
    height, width = frame.shape[:2]
    max_dim = 1000
    scale = 1.0
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    print("Press 's' to select ROIs on the current frame.")
    
    while True:
        cv2.imshow("Config Tool", frame)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            raw_rois = select_rois(frame)
            
            config = load_config()
            new_rois = []
            
            for rect in raw_rois:
                # Scale back to original coordinates if needed
                x, y, w, h = rect
                original_rect = [int(x/scale), int(y/scale), int(w/scale), int(h/scale)]
                
                label = get_label_for_roi(rect, frame)
                
                # Calculate Histogram for the ROI
                roi_img = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                # Calculate histogram for Hue and Saturation
                hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                hist_list = hist.flatten().tolist() # Convert to list for JSON serialization

                new_rois.append({
                    "label": label,
                    "rect": original_rect, # [x, y, w, h]
                    "histogram": hist_list
                })
            
            config["rois"].extend(new_rois)
            save_config(config)
            print("ROIs added to configuration.")
            break
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
