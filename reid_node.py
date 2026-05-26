import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['AV_LOG_FORCE_NOCOLOR'] = '1'
os.environ['AV_LOG_LEVEL'] = 'quiet'
import json
import base64
import tkinter as tk
from threading import Thread, Lock
from tkinter import simpledialog
import cv2
import time
import numpy as np
from dotenv import load_dotenv
from ultralytics import RTDETR
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 1. GAG THE FFMPEG & H264 SPAM
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# 2. SAFE CONNECTION FLAGS
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# DIRECTORY SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

load_dotenv()
rtsp_url = os.getenv("CAMERA_URL")
model = RTDETR("rtdetr-l.pt")

# OPENAI SCENE-ANALYSIS SETUP
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("[WARN] OPENAI_API_KEY not set in .env — the [D]escribe key will not work.")
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
SCENE_MODEL = "gpt-5.5"  # OpenAI flagship vision model (Responses API)

# GALLERY SETTINGS
MAX_SLOTS = 16
signature_gallery = [None] * MAX_SLOTS
slot_names = ["Empty"] * MAX_SLOTS
LABEL_FILE = os.path.join(BASE_DIR, "labels.json")

# SESSION MEMORY (Anti-False-Positive System)
track_identity_cache = {}
track_validation_history = {}
REQUIRED_MATCHES = 7

# SCENE-DESCRIPTION STATE (shared between main loop and worker thread)
scene_lock = Lock()
scene_description = ""
scene_status = "idle"   # idle | analyzing | done | error

root = tk.Tk()
root.withdraw()

# LOAD DATABASE
print("\n" + "=" * 40)
print("SYSTEM AUDIT: LOADING DATABASE")
print("=" * 40)
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
print("=" * 40 + "\n")


# ROCK-SOLID VIDEO THREAD (100% Non-Blocking)
class VideoStream:
    def __init__(self, url):
        self.url = url
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is None:
                self.stream = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.stream is not None and self.stream.isOpened():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.grabbed = False
                    self.stream.release()
                    self.stream = None
                    time.sleep(1)
                    continue
                self.frame = frame
                self.grabbed = True
            else:
                self.grabbed = False
                if self.stream is not None:
                    self.stream.release()
                self.stream = None
                time.sleep(1)

    def read(self):
        return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()


def get_advanced_signature(car_crop):
    car_crop = cv2.resize(car_crop, (128, 256))
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    structure = cv2.resize(gray, (16, 16)).flatten() / 255.0
    return np.hstack((hist.flatten(), structure)).reshape(1, -1)


# SCENE-DESCRIPTION HELPERS

def wrap_text(text, font, font_scale, thickness, max_width):
    """Greedy word-wrap so a long string fits in max_width pixels."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        candidate = (current + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if tw <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def describe_scene_worker(frame_snapshot):
    """Encode the frame, call OpenAI, and stash the answer for the renderer."""
    global scene_description, scene_status

    if openai_client is None:
        with scene_lock:
            scene_description = "OPENAI_API_KEY not configured in .env"
            scene_status = "error"
        return

    try:
        # JPEG-encode instead of PNG: smaller payload, faster upload.
        ok, buf = cv2.imencode(".jpg", frame_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            with scene_lock:
                scene_description = "Failed to JPEG-encode the frame."
                scene_status = "error"
            return
        b64 = base64.b64encode(buf).decode("utf-8")

        response = openai_client.responses.create(
            model=SCENE_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a surveillance assistant looking at a single frame "
                            "from a security camera. Describe the scene in 3-5 sentences. "
                            "Focus on: vehicles (color, make/type if identifiable, position, "
                            "direction of travel if inferable), people (count, posture, what "
                            "they appear to be doing), and any notable activity or "
                            "environmental context (lighting, weather, time of day). "
                            "Be specific, concrete, and concise. Do not speculate beyond "
                            "what is visible."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    },
                ],
            }],
        )
        text = (response.output_text or "").strip()
        with scene_lock:
            scene_description = text or "(empty response)"
            scene_status = "done"
        print(f"\n[SCENE @ {time.strftime('%H:%M:%S')}] {text}\n")
    except Exception as e:
        with scene_lock:
            scene_description = f"API error: {str(e)[:140]}"
            scene_status = "error"
        print(f"[SCENE-ERR] {e}")


# MAIN RUNTIME
vstream = VideoStream(rtsp_url).start()

while True:
    frame = vstream.read()

    # SAFE STANDBY LOGIC
    if frame is None:
        standby_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby_frame, "SYSTEM STANDBY", (140, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(standby_frame, "Camera offline. Waiting for motion...", (120, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
        cls_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, ids, cls_ids):
            x1, y1, x2, y2 = box

            # PERSON (COCO class 0): track and draw, but skip vehicle Re-ID entirely.
            # Checked first so a reused track_id can't carry a stale "LOCKED" car label
            # onto a person if the tracker recycles IDs after a timeout.
            if cls_id == 0:
                box_color, thickness = (255, 200, 0), 2
                label = f"PERSON ID:{track_id}"
            elif track_id in track_identity_cache:
                name, _ = track_identity_cache[track_id]
                box_color, thickness = (0, 215, 255), 3
                label = f"{name} (LOCKED)"
            else:
                car_crop = frame[max(0, y1):y2, max(0, x1):x2]
                if car_crop.size == 0:
                    continue
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
                            label = f"{name} ({int(avg_score * 100)}%)"
                        else:
                            track_validation_history[track_id].pop(0)
                            box_color, thickness = (0, 255, 255), 2
                            label = f"ANALYZING... (Avg: {int(avg_score * 100)}%)"
                    else:
                        box_color, thickness = (0, 255, 255), 2
                        label = f"ANALYZING... ({len(track_validation_history[track_id])}/{REQUIRED_MATCHES})"
                else:
                    track_validation_history[track_id] = []
                    box_color, thickness = (0, 0, 255), 2
                    label = f"UNKNOWN CAR ID:{track_id}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # 16-SLOT HUD
    cv2.rectangle(display_frame, (10, 10), (460, 180), (0, 0, 0), -1)
    cv2.putText(display_frame, "IDENTITY DATABASE:", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i in range(MAX_SLOTS):
        col, row = (0, i) if i < 8 else (1, i - 8)
        x_pos, y_pos = (20 if col == 0 else 240), (60 + (row * 15))
        color = (0, 255, 0) if signature_gallery[i] is not None else (100, 100, 100)
        cv2.putText(display_frame, f"S{i:02d}: {slot_names[i]}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # KEY HINT (top-right)
    hint = "[D] Describe Scene  [S] Save Vehicle  [Q] Quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(display_frame, hint,
                (display_frame.shape[1] - hw - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # SCENE ANALYSIS OVERLAY (bottom of frame)
    with scene_lock:
        status_now = scene_status
        desc_now = scene_description

    if status_now != "idle":
        h, w = display_frame.shape[:2]
        if status_now == "analyzing":
            shown_text = f"Analyzing scene with {SCENE_MODEL}..."
            header_color = (0, 255, 255)
        elif status_now == "done":
            shown_text = desc_now
            header_color = (0, 255, 0)
        else:
            shown_text = desc_now
            header_color = (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick, line_h = 0.55, 1, 22
        max_text_width = w - 40
        lines = wrap_text(shown_text, font, scale, thick, max_text_width)

        n = max(1, len(lines))
        header_offset = 32
        first_line_offset = header_offset + 24
        last_line_offset = first_line_offset + (n - 1) * line_h
        panel_h = last_line_offset + 14
        y_top = h - panel_h - 10

        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, y_top), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        cv2.rectangle(display_frame, (10, y_top), (w - 10, h - 10), header_color, 2)

        cv2.putText(display_frame, f"SCENE ANALYSIS ({SCENE_MODEL})",
                    (20, y_top + header_offset), font, 0.6, header_color, 2)
        for i, ln in enumerate(lines):
            y = y_top + first_line_offset + i * line_h
            cv2.putText(display_frame, ln, (20, y), font, scale, (255, 255, 255), thick)

    cv2.imshow("Enterprise Re-ID Node", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # ENROLL VEHICLE
    if key == ord('s') or key == ord('S'):
        if current_target_sig is not None:
            new_name = simpledialog.askstring("Enrollment", "Enter Name:")
            if new_name:
                slot = simpledialog.askinteger("Enrollment", "Select Slot (0-15):")
                if slot is not None and 0 <= slot < MAX_SLOTS:
                    np.save(os.path.join(BASE_DIR, f"sig_{slot}.npy"), current_target_sig)
                    signature_gallery[slot] = current_target_sig
                    slot_names[slot] = new_name
                    with open(LABEL_FILE, "w") as f:
                        json.dump(slot_names, f)
        else:
            print("No detection. Wait for a box.")

    # DESCRIBE CURRENT SCENE WITH GPT-5.5
    if key == ord('d') or key == ord('D'):
        with scene_lock:
            already_running = scene_status == "analyzing"
        if not already_running:
            snapshot = frame.copy()
            with scene_lock:
                scene_description = ""
                scene_status = "analyzing"
            Thread(target=describe_scene_worker, args=(snapshot,), daemon=True).start()

    if key == ord('q') or key == ord('Q'):
        vstream.stop()
        os._exit(0)
