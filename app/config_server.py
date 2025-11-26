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
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {CONFIG_FILE} is corrupted. Backing up and creating a new one.")
            os.rename(CONFIG_FILE, CONFIG_FILE + ".bak")
            return {"rois": []}
    return {"rois": []}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frame')
def get_frame():
    global VIDEO_PATH
    if VIDEO_PATH is None:
        return "Video not loaded", 400
    
    frame_idx = request.args.get('index', default=0, type=int)
    
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    # Use absolute path matching main()
    cache_dir = os.path.join(os.getcwd(), "frames_cache", video_name)
    frame_path = os.path.join(cache_dir, f"frame_{frame_idx}.jpg")
    
    if os.path.exists(frame_path):
        return send_file(frame_path, mimetype='image/jpeg')
    else:
        # Fallback if frame missing (or end of video loop logic handled by frontend requesting 0)
        # If index is out of bounds, maybe return 404 or loop?
        # Let's try to return frame 0 if out of bounds to loop
        frame_0 = os.path.join(cache_dir, "frame_0.jpg")
        if os.path.exists(frame_0):
             return send_file(frame_0, mimetype='image/jpeg')
        return "Frame not found", 404

@app.route('/video_info')
def get_video_info():
    global CAP
    if CAP is None:
        return "Video not loaded", 400
    
    total_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = CAP.get(cv2.CAP_PROP_FPS)
    
    return jsonify({
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps
    })

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
    import re
    # Sanitize label for filename
    safe_label = re.sub(r'[^a-zA-Z0-9]', '_', data.get('label', 'unknown'))
    image_filename = f"roi_bottle_{safe_label}.jpg"
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
    
    # Frame Caching Logic
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    # Use absolute path for cache to avoid CWD issues with Flask
    cache_dir = os.path.join(os.getcwd(), "frames_cache", video_name)
    
    if not os.path.exists(cache_dir):
        print(f"Extracting frames to cache: {cache_dir} ... This might take a while.")
        os.makedirs(cache_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_count = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save as jpg
            cv2.imwrite(os.path.join(cache_dir, f"frame_{frame_count}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Extracted {frame_count}/{total} frames...")
        
        cap.release()
        print("Frame extraction complete.")
    else:
        print(f"Using existing frame cache: {cache_dir}")

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
                                import re
                                safe_label = re.sub(r'[^a-zA-Z0-9]', '_', roi['label'])
                                image_filename = f"roi_bottle_{safe_label}.jpg"
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
