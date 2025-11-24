import cv2
import json
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import io

app = Flask(__name__)

CONFIG_FILE = 'bottle_config.json'
VIDEO_PATH = None
CAP = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"rois": []}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frame')
def get_frame():
    global CAP
    if CAP is None:
        return "Video not loaded", 400
    
    # Get specific frame index if requested, else current
    frame_idx = request.args.get('index', type=int)
    if frame_idx is not None:
        CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = CAP.read()
    if not ret:
        # Loop back to start if end reached
        CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = CAP.read()
    
    if not ret:
        return "Could not read frame", 500

    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(load_config())
    else:
        config = request.json
        save_config(config)
        return jsonify({"status": "saved"})

@app.route('/histogram', methods=['POST'])
def calculate_histogram():
    global CAP
    if CAP is None:
        return "Video not loaded", 400

    data = request.json
    points = data.get('points') # List of [x, y]
    frame_idx = data.get('frame_index', 0) # Optional, to ensure we use the right frame
    
    # We might need to seek to the frame again if the frontend sends an index
    # For simplicity, we assume the frontend is showing the current frame or we just use the last read frame
    # Better approach: Frontend sends the frame index it is looking at.
    
    CAP.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = CAP.read()
    if not ret:
        return "Could not read frame", 500

    # Create mask from polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Calculate histogram
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist_list = hist.flatten().tolist()
    
    # Save ROI image for reference
    # Crop the bounding rect of the polygon
    x, y, w, h = cv2.boundingRect(pts)
    roi_crop = frame[y:y+h, x:x+w].copy()
    
    # Create a mask for the crop (optional, to show only the polygon area)
    # crop_mask = np.zeros(roi_crop.shape[:2], dtype=np.uint8)
    # pts_shifted = pts - [x, y]
    # cv2.fillPoly(crop_mask, [pts_shifted], 255)
    # roi_crop = cv2.bitwise_and(roi_crop, roi_crop, mask=crop_mask)
    
    import time
    timestamp = int(time.time() * 1000)
    image_filename = f"roi_{timestamp}.jpg"
    image_path = os.path.join("roi_images", image_filename)
    cv2.imwrite(image_path, roi_crop)
    
    return jsonify({
        "histogram": hist_list,
        "image_path": image_path
    })

def main():
    global VIDEO_PATH, CAP
    if len(sys.argv) < 2:
        print("Usage: python config_server.py <video_path>")
        sys.exit(1)
    
    VIDEO_PATH = sys.argv[1]
    CAP = cv2.VideoCapture(VIDEO_PATH)
    if not CAP.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        sys.exit(1)
        
    print(f"Starting server for video: {VIDEO_PATH}")
    
    # Retroactive ROI Image Generation
    config = load_config()
    rois_updated = False
    
    if config["rois"]:
        # Check if any ROI is missing image_path or file
        missing_images = False
        for roi in config["rois"]:
            if "image_path" not in roi or not os.path.exists(roi["image_path"]):
                missing_images = True
                break
        
        if missing_images:
            print("Generating missing ROI reference images...")
            # Read a frame (e.g., frame 0)
            CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = CAP.read()
            
            if ret:
                import time
                if not os.path.exists("roi_images"):
                    os.makedirs("roi_images")
                    
                for i, roi in enumerate(config["rois"]):
                    if "image_path" not in roi or not os.path.exists(roi["image_path"]):
                        if "points" in roi:
                            pts = np.array(roi['points'], dtype=np.int32)
                            x, y, w, h = cv2.boundingRect(pts)
                            # Ensure crop is within frame
                            x, y = max(0, x), max(0, y)
                            w = min(w, frame.shape[1] - x)
                            h = min(h, frame.shape[0] - y)
                            
                            if w > 0 and h > 0:
                                roi_crop = frame[y:y+h, x:x+w].copy()
                                timestamp = int(time.time() * 1000) + i
                                image_filename = f"roi_retro_{timestamp}.jpg"
                                image_path = os.path.join("roi_images", image_filename)
                                cv2.imwrite(image_path, roi_crop)
                                roi["image_path"] = image_path
                                rois_updated = True
                                print(f"Generated image for ROI: {roi['label']}")
            
            if rois_updated:
                save_config(config)
                print("Config updated with new ROI images.")

    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()
