import cv2
import numpy as np
import os
import re
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class InteractionLogic:
    def __init__(self, config):
        self.config = config
        self.rois = config.get("rois", [])
        
        # State tracking
        # key: label, value: { "last_seen": timestamp, "dwell_score": float (0-100) }
        self.roi_states = {roi['label']: {"last_seen": 0, "dwell_score": 0} for roi in self.rois if 'label' in roi}
        
        # Load Classification Model for Vector Similarity
        try:
            print("Loading ResNet18 for similarity matching...")
            self.model = models.resnet18(weights='DEFAULT')
            self.model.fc = torch.nn.Identity() # Remove classification layer to get embeddings
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.use_vector_sim = True
        except Exception as e:
            print(f"Warning: Could not load ResNet18: {e}. Fallback to Histogram.")
            self.use_vector_sim = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Pre-compute embeddings for ROIs
        for roi in self.rois:
            roi["ref_img"] = None
            roi["embedding"] = None
            roi["hist_np"] = None
            
            # Load Label Image
            safe_label = re.sub(r'[^a-zA-Z0-9]', '_', roi['label'])
            image_filename = f"roi_bottle_{safe_label}.jpg"
            
            paths_to_check = [
                os.path.join("roi_images", image_filename),
                os.path.join("..", "roi_images", image_filename)
            ]
            
            for path in paths_to_check:
                if os.path.exists(path):
                    roi["ref_img"] = cv2.imread(path)
                    break
            
            if roi["ref_img"] is not None:
                # Compute Vector Embedding
                if self.use_vector_sim:
                    roi["embedding"] = self._compute_embedding(roi["ref_img"])
                
                # Compute Histogram (Fallback)
                hsv_img = cv2.cvtColor(roi["ref_img"], cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                roi["hist_np"] = hist.reshape((180, 256))
            else:
                print(f"Warning: No valid reference image for {roi['label']}")

    def _compute_embedding(self, cv2_img):
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        return embedding.cpu().numpy().flatten()

    def _cosine_similarity(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def is_point_in_roi(self, point, roi):
        px, py = point
        if 'points' in roi:
            pts = np.array(roi['points'], dtype=np.int32)
            return cv2.pointPolygonTest(pts, (px, py), False) >= 0
        elif 'rect' in roi:
            rx, ry, rw, rh = roi['rect']
            return rx <= px <= rx + rw and ry <= py <= ry + rh
        return False

    def check_hand_in_roi(self, hand_box):
        # hand_box might have 5 elements now (x1, y1, x2, y2, track_id)
        hx1, hy1, hx2, hy2 = hand_box[:4] 
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2
        
        for roi in self.rois:
            if self.is_point_in_roi((h_cx, h_cy), roi):
                return roi['label']
        return None

    def check_roi_occupancy(self, hands_list):
        # Identify which ROIs are 'active' (have a hand)
        # We removed bottle detections from shelves as per new requirement
        active_rois = set()
        
        # Check Hands (Hands keep ROI active)
        for hand in hands_list:
            hx1, hy1, hx2, hy2 = hand[:4]
            h_center = ((hx1+hx2)//2, (hy1+hy2)//2)
            for roi in self.rois:
                if self.is_point_in_roi(h_center, roi):
                    active_rois.add(roi['label'])
                    
        return active_rois

    def update_state(self, hand_in_roi_label, current_time=None):
        if current_time is None:
            current_time = time.time()
            
        # Update dwell times
        for label, state in self.roi_states.items():
            if label == hand_in_roi_label:
                # User is interacting NOW
                state["last_seen"] = current_time
                state["dwell_score"] = min(100, state["dwell_score"] + 5) # Faster increase
            else:
                # Decay logic
                time_diff = current_time - state["last_seen"]
                
                if time_diff < 5.0:
                    # Valid window (0-5s), maintain score? Or slight decay?
                    # Request says "valid during approx 5 secs", lets keep it steady or slight decay
                    pass 
                elif 5.0 <= time_diff < 10.0:
                    # Linear Decay from 5s to 10s
                    # At 5s -> factor 1.0
                    # At 10s -> factor 0.0
                    decay_factor = 1.0 - ((time_diff - 5.0) / 5.0)
                    state["dwell_score"] *= decay_factor
                else:
                    # > 10s, invalid
                    state["dwell_score"] = 0

    def find_closest_bottle(self, hand_box, bottle_detections):
        if not bottle_detections:
            return None
        hx1, hy1, hx2, hy2 = hand_box[:4]
        h_cx, h_cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        
        min_dist = float('inf')
        closest_bottle = None
        
        for bottle in bottle_detections:
            bx1, by1, bx2, by2 = bottle['bbox']
            b_cx, b_cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            dist = np.sqrt((h_cx - b_cx)**2 + (h_cy - b_cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_bottle = bottle
        return closest_bottle

    def get_annotated_scores(self, bottle_crop, active_rois):
        # active_rois: set of labels that potentially contain something
        
        if bottle_crop.size == 0:
            return {"visual_scores": [], "final_scores": []}
            
        final_ranking = []
        visual_ranking = []
        
        # 1. Compute Visual Similarity
        visual_sims = {}
        if self.use_vector_sim:
            crop_emb = self._compute_embedding(bottle_crop)
            for roi in self.rois:
                if roi["embedding"] is not None:
                    sim = self._cosine_similarity(crop_emb, roi["embedding"])
                    # Visual sim is usually 0.7-1.0 for same objects with ResNet
                    # We normalize it loosely to 0-1 range for scoring
                    visual_sims[roi['label']] = max(0, sim)
                else:
                    visual_sims[roi['label']] = 0.0
        else:
            # Fallback Histogram
            hsv_crop = cv2.cvtColor(bottle_crop, cv2.COLOR_BGR2HSV)
            hist_crop = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist_crop, hist_crop, 0, 255, cv2.NORM_MINMAX)
            flat_hist_crop = hist_crop.reshape((180, 256))
            
            for roi in self.rois:
                if roi["hist_np"] is not None:
                    sim = cv2.compareHist(flat_hist_crop, roi["hist_np"], cv2.HISTCMP_CORREL)
                    visual_sims[roi['label']] = max(0, sim)
                else:
                    visual_sims[roi['label']] = 0.0

        # 2. Combine with Logical Factors
        for roi in self.rois:
            label = roi['label']
            
            # Factor A: Visual Similarity (Base)
            vis_score = visual_sims.get(label, 0)
            
            visual_ranking.append({
                "label": label,
                "score": vis_score,
                "ref_img": roi.get("ref_img")
            })
            
            # Factor B: ROI Occupancy (Hard Filter)
            # If ROI is empty (no bottle, no hand), it shouldn't match
            score = vis_score
            if label not in active_rois:
                score *= 0.1 # Heavily penalize
            
            # Factor C: Dwell Time / Priority
            # User said: "Should take into account where hand spent most time"
            dwell_state = self.roi_states.get(label, {"dwell_score": 0})
            dwell_bonus = (dwell_state["dwell_score"] / 100.0) * 0.4 # Bonus up to 0.4
            
            final_score = score + dwell_bonus
            
            final_ranking.append({
                "label": label,
                "score": final_score,
                "ref_img": roi.get("ref_img"),
                "visual_sim": vis_score,
                "dwell_val": dwell_state["dwell_score"]
            })
            
        visual_ranking.sort(key=lambda x: x["score"], reverse=True)
        final_ranking.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "visual_scores": visual_ranking[:5],
            "final_scores": final_ranking[:5]
        }
