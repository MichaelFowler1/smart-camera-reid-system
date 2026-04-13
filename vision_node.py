import os
import json
import tkinter as tk
from threading import Thread
from tkinter import simpledialog

# 1. GAG THE FFMPEG & H264 SPAM
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# 2. SAFE CONNECTION FLAGS
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

import cv2
import time
import numpy as np
from dotenv import load_dotenv
from ultralytics import RTDETR
from sklearn.metrics.pairwise import cosine_similarity

# DIRECTORY SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

load_dotenv()
rtsp_url = os.getenv("CAMERA_URL")
model = RTDETR("rtdetr-l.pt")

# GALLERY SETTINGS
MAX_SLOTS = 16
signature_gallery = [None] * MAX_SLOTS
slot_names = ["Empty"] * MAX_SLOTS 
LABEL_FILE = os.path.join(BASE_DIR, "labels.json")

# SESSION MEMORY (Anti-False-Positive System)
track_identity_cache = {} 
track_validation_history = {} 
REQUIRED_MATCHES = 7 

root = tk.Tk()
root.withdraw()

# LOAD DATABASE
print("\n" + "="*40)
print("SYSTEM AUDIT: LOADING DATABASE")
print("="*40)
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
        slot_names = json.load(f)
        if len(slot_names) < MAX_SLOTS:
            slot_names.extend(["Empty"] * (MAX_SLOTS - len(slot_names)))

for i in range(MAX_SLOTS):
    filename = os.path.join(BASE_DIR, f"sig_{i}.npy")
    if os.path.exists(filename):
        sig = np.load(filename)
        if sig.shape[1] == 768:
            signature_gallery[i] = sig
            print(f"[LOADED] Slot {i:02d}: {slot_names[i]}")
print("="*40 + "\n")

# ROCK-SOLID VIDEO THREAD (No Race Conditions)
class VideoStream:
    def __init__(self, url):
        self.url = url
        self.stream = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.grabbed = False # Safely tell main thread we are offline
                self.stream.release()
                time.sleep(2) # Wait 2 seconds before knocking on Eufy's door again
                self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            self.frame = frame
            self.grabbed = True

    def read(self):
        # Only return the frame if we successfully grabbed it, otherwise return None
        return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()

def get_advanced_signature(car_crop):
    car_crop = cv2.resize(car_crop, (128, 256))
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    structure = cv2.resize(gray, (16, 16)).flatten() / 255.0 
    return np.hstack((hist.flatten(), structure)).reshape(1, -1)

# MAIN RUNTIME
vstream = VideoStream(rtsp_url).start()

while True:
    frame = vstream.read()
    
    # SAFE STANDBY LOGIC
    if frame is None:
        standby_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby_frame, "SYSTEM STANDBY", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(standby_frame, "Camera offline. Waiting for motion...", (120, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Enterprise Re-ID Node", standby_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            vstream.stop()
            os._exit(0)
        continue 
    
    display_frame = frame.copy()

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, 
                          device=0, classes=[2, 7], conf=0.8, half=True, verbose=False)

    current_target_sig = None

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            if track_id in track_identity_cache:
                name, _ = track_identity_cache[track_id]
                box_color, thickness = (0, 215, 255), 3 
                label = f"{name} (LOCKED)"
            else:
                car_crop = frame[max(0, y1):y2, max(0, x1):x2]
                if car_crop.size == 0: continue
                current_target_sig = get_advanced_signature(car_crop)
                
                highest_sim, best_idx = 0, -1
                for i, saved_sig in enumerate(signature_gallery):
                    if saved_sig is not None:
                        sim = cosine_similarity(current_target_sig, saved_sig)[0][0]
                        if sim > highest_sim:
                            highest_sim, best_idx = sim, i
                
                if highest_sim > 0.80: 
                    if track_id not in track_validation_history:
                        track_validation_history[track_id] = []
                    
                    track_validation_history[track_id].append((highest_sim, best_idx))
                    
                    if len(track_validation_history[track_id]) >= REQUIRED_MATCHES:
                        recent_scores = [s for s, idx in track_validation_history[track_id][-REQUIRED_MATCHES:]]
                        recent_idxs = [idx for s, idx in track_validation_history[track_id][-REQUIRED_MATCHES:]]
                        
                        avg_score = sum(recent_scores) / len(recent_scores)
                        most_common_idx = max(set(recent_idxs), key=recent_idxs.count)
                        
                        if avg_score > 0.83:
                            name = slot_names[most_common_idx]
                            track_identity_cache[track_id] = (name, avg_score)
                            box_color, thickness = (0, 215, 255), 3 
                            label = f"{name} ({int(avg_score*100)}%)"
                        else:
                            track_validation_history[track_id].pop(0)
                            box_color, thickness = (0, 255, 255), 2 
                            label = f"ANALYZING... (Avg: {int(avg_score*100)}%)"
                    else:
                        box_color, thickness = (0, 255, 255), 2 
                        label = f"ANALYZING... ({len(track_validation_history[track_id])}/{REQUIRED_MATCHES})"
                else:
                    track_validation_history[track_id] = []
                    box_color, thickness = (0, 0, 255), 2 
                    label = f"UNKNOWN CAR ID:{track_id}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # 16-SLOT HUD
    cv2.rectangle(display_frame, (10, 10), (460, 180), (0, 0, 0), -1)
    cv2.putText(display_frame, "IDENTITY DATABASE:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i in range(MAX_SLOTS):
        col, row = (0, i) if i < 8 else (1, i - 8)
        x_pos, y_pos = (20 if col == 0 else 240), (60 + (row * 15))
        color = (0, 255, 0) if signature_gallery[i] is not None else (100, 100, 100)
        cv2.putText(display_frame, f"S{i:02d}: {slot_names[i]}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imshow("Enterprise Re-ID Node", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        if current_target_sig is not None:
            new_name = simpledialog.askstring("Enrollment", "Enter Name:")
            if new_name:
                slot = simpledialog.askinteger("Enrollment", "Select Slot (0-15):")
                if slot is not None and 0 <= slot < MAX_SLOTS:
                    np.save(os.path.join(BASE_DIR, f"sig_{slot}.npy"), current_target_sig)
                    signature_gallery[slot] = current_target_sig
                    slot_names[slot] = new_name
                    with open(LABEL_FILE, "w") as f: json.dump(slot_names, f)
        else:
            print("No detection. Wait for a box.")

    if key == ord('q') or key == ord('Q'):
        vstream.stop()
        os._exit(0)