import numpy as np
from ultralytics import YOLO
import cv2

CONF_THRESHOLD = 0.4

# class BottleDetector:
#     def __init__(self, model_path):
#         print(f"Loading YOLO model from {model_path}...")
#         self.model = YOLO(model_path)
#         
#     def detect(self, frame):
#         # Run inference with ByteTrack
#         # persist=True is important for tracking
#         results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
#         detections = []
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 b = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
#                 conf = box.conf[0].cpu().numpy()
#                 cls = box.cls[0].cpu().numpy()
#                 track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
#                 
#                 if conf > CONF_THRESHOLD:
#                     detections.append({
#                         "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
#                         "conf": float(conf),
#                         "class": int(cls),
#                         "track_id": track_id
#                     })
#         return detections

class BottleSegmenter:
    def __init__(self, model_path='/home/manuamest/Repos/ShelfPickVA/runs/segment/bottle_seg_v15/weights/best.pt'):
        print(f"Loading YOLO Segmentation model from {model_path}...")
        self.model = YOLO(model_path)
        
    def segment_bottle(self, crop_image):
        if crop_image.size == 0:
            return None, None
            
        # Detect bottles (Custom model has only class 0: 'Bottle in hand')
        results = self.model(crop_image, verbose=False, classes=[0])
        
        largest_mask = None
        masked_img = None
        max_area = 0
        
        for r in results:
            if r.masks is None: continue
            
            # Iterate through masks
            for i, mask in enumerate(r.masks.data):
                # mask is (H, W) tensor on GPU or CPU
                m = mask.cpu().numpy().astype('uint8') * 255
                m_resized = cv2.resize(m, (crop_image.shape[1], crop_image.shape[0]))
                
                area = np.sum(m_resized > 0)
                if area > max_area:
                    max_area = area
                    largest_mask = m_resized
                    
        if largest_mask is not None:
            # Create masked image (black background)
            masked_img = cv2.bitwise_and(crop_image, crop_image, mask=largest_mask)
            return masked_img, largest_mask
            
        return None, None

class HandDetector:
    def __init__(self, model_path):
        print(f"Loading YOLO Pose model from {model_path}...")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        # Use tracking instead of simple inference
        results = self.model.track(frame, persist=True, verbose=False)
        hands_list = []
        
        for r in results:
            keypoints = r.keypoints
            boxes = r.boxes
            if keypoints is None: continue
            
            # Keypoints data is (N, 17, 3)
            # Boxes IDs are (N,)
            
            for i, kps in enumerate(keypoints.data):
                # Get Track ID if available
                track_id = -1
                if boxes is not None and boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i].cpu().numpy())

                # kps is (17, 3) -> x, y, conf
                # Wrist indices: 9 (left), 10 (right)
                for wrist_idx in [9, 10]:
                    if kps[wrist_idx][2] > 0.5: # Confidence check
                        wx, wy = int(kps[wrist_idx][0]), int(kps[wrist_idx][1])
                        
                        # Estimate hand box around wrist
                        box_size = 60
                        x_min = max(0, wx - box_size)
                        y_min = max(0, wy - box_size)
                        x_max = min(frame.shape[1], wx + box_size)
                        y_max = min(frame.shape[0], wy + box_size)
                        
                        # Add track_id to the list [x1, y1, x2, y2, track_id]
                        hands_list.append([x_min, y_min, x_max, y_max, track_id])
                        
        return hands_list
