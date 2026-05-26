"""
doorbell_node.py
----------------
Stripped-down doorbell vision pipeline. Reads the UDP MPEG-TS stream
emitted by eufy_bridge_doorbell.py and runs:

  - Person detection via RT-DETR (COCO class 0 only — no vehicle Re-ID)
  - Face detection + identification via facenet-pytorch on CUDA
    (MTCNN finds faces, InceptionResnetV1 produces 512-dim embeddings,
    cosine similarity against an enrolled gallery)
  - GPT-5.5 scene description on [D] key (doorbell-tuned prompt)
  - Face enrollment on [S] key — saves the largest currently-visible
    face into a numbered slot (0-15)

Gallery files (created on enrollment, stored next to this script):
    face_sig_NN.npy     # 512-dim L2-normalized embedding
    face_labels.json    # list of 16 slot names

One-time deps:
    py -3.12 -m pip install facenet-pytorch pillow

(You already have torch+CUDA, opencv-python, ultralytics, openai,
 python-dotenv, scikit-learn.)

.env additions:
    EUFY_DOORBELL_SERIAL=T8200N0020281CF1
    DOORBELL_URL=udp://127.0.0.1:5001?fifo_size=5000000&overrun_nonfatal=1

First run: facenet-pytorch downloads ~110MB of VGGFace2 weights into
the torch hub cache. Just lets it happen.

Run order (after the camera stack):
    Window 1: eufy-security-server --config config.json     (shared)
    Window 4: py -3.12 eufy_bridge_doorbell.py
    Window 5: py -3.12 doorbell_node.py
"""

import os

# Silence ffmpeg/libav startup noise before anything imports cv2.
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['AV_LOG_FORCE_NOCOLOR'] = '1'
os.environ['AV_LOG_LEVEL'] = 'quiet'

import base64
import json
import time
import tkinter as tk
from threading import Thread, Lock
from tkinter import simpledialog

import cv2
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from facenet_pytorch import MTCNN, InceptionResnetV1
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import RTDETR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
DOORBELL_URL = os.getenv("DOORBELL_URL")
if not DOORBELL_URL:
    raise SystemExit(
        "Set DOORBELL_URL in .env, e.g.\n"
        "  DOORBELL_URL=udp://127.0.0.1:5001?fifo_size=5000000&overrun_nonfatal=1"
    )

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("[WARN] OPENAI_API_KEY not set — [D] key (scene description) will fail.")
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
SCENE_MODEL = "gpt-5.5"

# Face recognition setup ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] face recognition device = {device}")

# select_largest=True picks the dominant face in the crop, which for a
# doorbell view is almost always the person at the door.
mtcnn = MTCNN(
    image_size=160, margin=20, keep_all=False, select_largest=True,
    min_face_size=40, post_process=True, device=device,
)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
FACE_EMBED_DIM = 512

# Detection model -----------------------------------------------------------
det_model = RTDETR("rtdetr-l.pt")

# Face gallery --------------------------------------------------------------
MAX_FACE_SLOTS = 16
face_gallery = [None] * MAX_FACE_SLOTS
face_names = ["Empty"] * MAX_FACE_SLOTS
FACE_LABEL_FILE = os.path.join(BASE_DIR, "face_labels.json")

print("\n" + "=" * 40)
print("DOORBELL: LOADING FACE DATABASE")
print("=" * 40)
if os.path.exists(FACE_LABEL_FILE):
    with open(FACE_LABEL_FILE, "r") as f:
        face_names = json.load(f)
        if len(face_names) < MAX_FACE_SLOTS:
            face_names.extend(["Empty"] * (MAX_FACE_SLOTS - len(face_names)))

for i in range(MAX_FACE_SLOTS):
    fp = os.path.join(BASE_DIR, f"face_sig_{i}.npy")
    if os.path.exists(fp):
        emb = np.load(fp)
        if emb.shape[-1] == FACE_EMBED_DIM:
            face_gallery[i] = emb.reshape(1, -1) if emb.ndim == 1 else emb
            print(f"[LOADED] Slot {i:02d}: {face_names[i]}")
print("=" * 40 + "\n")

# Identification thresholds -------------------------------------------------
# Per-frame entry threshold (above this we start accumulating evidence).
ENTER_THRESH = 0.50
# Average over recent matches must exceed this to lock identity.
LOCK_THRESH = 0.60
# Number of consecutive high-similarity hits before locking a track.
REQUIRED_FACE_MATCHES = 3

# Tracking state ------------------------------------------------------------
track_identity_cache = {}        # track_id -> (name, avg_score)
track_face_history = {}          # track_id -> [(score, slot_idx), ...]

# Scene description state ---------------------------------------------------
scene_lock = Lock()
scene_description = ""
scene_status = "idle"

# Currently-detected face that S would enroll
current_face_embedding = None

# Tkinter for enrollment dialog
root = tk.Tk()
root.withdraw()


# ---------------------------------------------------------------------------
# Video stream (same pattern as reid_node.py — daemon thread reading UDP)

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


# ---------------------------------------------------------------------------
# Face recognition helper

def extract_face(person_crop_bgr):
    """
    Given a BGR crop of a person, return (embedding_1x512, face_bbox_xyxy_in_crop)
    or (None, None) if no usable face was found. The bbox is relative to
    the input crop, not the full frame.
    """
    if person_crop_bgr is None or person_crop_bgr.size == 0:
        return None, None
    h, w = person_crop_bgr.shape[:2]
    if h < 80 or w < 50:
        return None, None  # too small to find a face

    rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # 1) Get aligned, normalized face tensor (None if no face).
    try:
        face_tensor = mtcnn(pil)
    except Exception:
        return None, None
    if face_tensor is None:
        return None, None

    # 2) Get the bbox separately for visualization.
    try:
        boxes, probs = mtcnn.detect(pil)
    except Exception:
        boxes, probs = None, None
    if boxes is None or len(boxes) == 0:
        return None, None
    # mtcnn.detect with select_largest=True returns the largest first.
    bbox = boxes[0]
    prob = probs[0] if probs is not None else 1.0
    if prob is None or prob < 0.90:
        return None, None

    # 3) Compute the embedding.
    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()
    # facenet-pytorch with post_process=True returns L2-normalized vectors,
    # so cosine similarity == dot product.
    return emb, bbox


# ---------------------------------------------------------------------------
# Scene description (GPT-5.5)

def wrap_text(text, font, scale, thick, max_width):
    lines, current = [], ""
    for w in text.split():
        candidate = (current + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thick)
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
    global scene_description, scene_status
    if openai_client is None:
        with scene_lock:
            scene_description = "OPENAI_API_KEY not configured."
            scene_status = "error"
        return
    try:
        ok, buf = cv2.imencode(".jpg", frame_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            with scene_lock:
                scene_description = "Failed to encode frame."
                scene_status = "error"
            return
        b64 = base64.b64encode(buf).decode("utf-8")

        response = openai_client.responses.create(
            model=SCENE_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "You are analyzing a single frame from a smart doorbell "
                        "camera. Describe who is at the door and what they are doing "
                        "in 3-5 sentences. Focus on: number of people, what they're "
                        "wearing, what they're holding (packages, mail, tools, etc.), "
                        "any uniform or apparent role (delivery, mail carrier, "
                        "visitor, etc.), the action they appear to be taking, and "
                        "any context like time of day or weather. Be concrete and "
                        "concise. Do not speculate beyond what is visible."
                    )},
                    {"type": "input_image",
                     "image_url": f"data:image/jpeg;base64,{b64}"},
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


# ---------------------------------------------------------------------------
# Main runtime

vstream = VideoStream(DOORBELL_URL).start()

while True:
    frame = vstream.read()

    if frame is None:
        standby = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby, "DOORBELL STANDBY", (130, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(standby, "Waiting for stream from bridge...", (130, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Doorbell Node", standby)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            vstream.stop()
            os._exit(0)
        continue

    display = frame.copy()

    results = det_model.track(
        frame, persist=True, tracker="bytetrack.yaml", imgsz=640,
        device=0, classes=[0], conf=0.5, half=True, verbose=False,
    )

    current_face_embedding = None

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()

        # Sort by person box area, largest first — face recog gets the closest
        # person's frame budget first, and the "current face for enrollment"
        # is naturally the closest face.
        sized = sorted(
            zip(boxes, ids),
            key=lambda bi: (bi[0][2] - bi[0][0]) * (bi[0][3] - bi[0][1]),
            reverse=True,
        )

        for box, track_id in sized:
            x1, y1, x2, y2 = box
            face_box_to_draw = None
            face_box_color = None

            if track_id in track_identity_cache:
                name, avg = track_identity_cache[track_id]
                box_color, thickness = (0, 215, 255), 3
                label = f"{name} (LOCKED {int(avg*100)}%)"
            else:
                person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                emb, face_bbox = extract_face(person_crop)

                if emb is None:
                    box_color, thickness = (255, 200, 0), 2
                    label = f"PERSON ID:{track_id} (no face)"
                else:
                    # Hold the largest-face embedding for [S] enrollment.
                    if current_face_embedding is None:
                        current_face_embedding = emb

                    # Match against gallery
                    highest_sim, best_idx = 0.0, -1
                    for i, saved in enumerate(face_gallery):
                        if saved is not None:
                            sim = float(cosine_similarity(emb, saved)[0][0])
                            if sim > highest_sim:
                                highest_sim, best_idx = sim, i

                    # Translate face bbox from crop-space to frame-space
                    fx1 = x1 + int(max(0, face_bbox[0]))
                    fy1 = y1 + int(max(0, face_bbox[1]))
                    fx2 = x1 + int(face_bbox[2])
                    fy2 = y1 + int(face_bbox[3])
                    face_box_to_draw = (fx1, fy1, fx2, fy2)

                    if highest_sim > ENTER_THRESH:
                        track_face_history.setdefault(track_id, []).append(
                            (highest_sim, best_idx)
                        )
                        hist = track_face_history[track_id]
                        if len(hist) >= REQUIRED_FACE_MATCHES:
                            recent = hist[-REQUIRED_FACE_MATCHES:]
                            avg = sum(s for s, _ in recent) / len(recent)
                            idxs = [i for _, i in recent]
                            most_common_idx = max(set(idxs), key=idxs.count)
                            if avg > LOCK_THRESH:
                                name = face_names[most_common_idx]
                                track_identity_cache[track_id] = (name, avg)
                                box_color, thickness = (0, 215, 255), 3
                                label = f"{name} ({int(avg*100)}%)"
                                face_box_color = (0, 255, 0)
                            else:
                                hist.pop(0)
                                box_color, thickness = (0, 255, 255), 2
                                label = f"ANALYZING... avg {int(avg*100)}%"
                                face_box_color = (0, 255, 255)
                        else:
                            box_color, thickness = (0, 255, 255), 2
                            label = f"ANALYZING ({len(hist)}/{REQUIRED_FACE_MATCHES})"
                            face_box_color = (0, 255, 255)
                    else:
                        track_face_history[track_id] = []
                        box_color, thickness = (0, 100, 255), 2
                        label = f"UNKNOWN PERSON ID:{track_id}"
                        face_box_color = (0, 100, 255)

            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            if face_box_to_draw is not None:
                fx1, fy1, fx2, fy2 = face_box_to_draw
                cv2.rectangle(display, (fx1, fy1), (fx2, fy2), face_box_color, 2)

    # FACE DATABASE HUD (top-left) ------------------------------------------
    cv2.rectangle(display, (10, 10), (300, 180), (0, 0, 0), -1)
    cv2.putText(display, "FACE DATABASE:", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i in range(MAX_FACE_SLOTS):
        col, row = (0, i) if i < 8 else (1, i - 8)
        x_pos, y_pos = (20 if col == 0 else 160), (60 + (row * 15))
        color = (0, 255, 0) if face_gallery[i] is not None else (100, 100, 100)
        cv2.putText(display, f"F{i:02d}: {face_names[i][:10]}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Key hint (top-right) --------------------------------------------------
    hint = "[D] Describe  [S] Enroll Face  [Q] Quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(display, hint, (display.shape[1] - hw - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Scene overlay ---------------------------------------------------------
    with scene_lock:
        status_now = scene_status
        desc_now = scene_description

    if status_now != "idle":
        h, w = display.shape[:2]
        if status_now == "analyzing":
            shown = f"Analyzing scene with {SCENE_MODEL}..."
            header_color = (0, 255, 255)
        elif status_now == "done":
            shown = desc_now
            header_color = (0, 255, 0)
        else:
            shown = desc_now
            header_color = (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick, line_h = 0.55, 1, 22
        lines = wrap_text(shown, font, scale, thick, w - 40)
        n = max(1, len(lines))
        header_offset = 32
        first_line_offset = header_offset + 24
        last_line_offset = first_line_offset + (n - 1) * line_h
        panel_h = last_line_offset + 14
        y_top = h - panel_h - 10

        overlay = display.copy()
        cv2.rectangle(overlay, (10, y_top), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        cv2.rectangle(display, (10, y_top), (w - 10, h - 10), header_color, 2)
        cv2.putText(display, f"SCENE ANALYSIS ({SCENE_MODEL})",
                    (20, y_top + header_offset), font, 0.6, header_color, 2)
        for i, ln in enumerate(lines):
            cv2.putText(display, ln,
                        (20, y_top + first_line_offset + i * line_h),
                        font, scale, (255, 255, 255), thick)

    cv2.imshow("Doorbell Node", display)
    key = cv2.waitKey(1) & 0xFF

    # ENROLL FACE -----------------------------------------------------------
    if key in (ord("s"), ord("S")):
        if current_face_embedding is not None:
            new_name = simpledialog.askstring("Face Enrollment", "Enter Name:")
            if new_name:
                slot = simpledialog.askinteger("Face Enrollment",
                                               "Select Slot (0-15):")
                if slot is not None and 0 <= slot < MAX_FACE_SLOTS:
                    np.save(os.path.join(BASE_DIR, f"face_sig_{slot}.npy"),
                            current_face_embedding)
                    face_gallery[slot] = current_face_embedding
                    face_names[slot] = new_name
                    with open(FACE_LABEL_FILE, "w") as f:
                        json.dump(face_names, f)
                    print(f"[enrolled] {new_name} -> slot {slot}")
        else:
            print("[enroll] No face currently visible. Get closer / face the camera.")

    # SCENE DESCRIBE --------------------------------------------------------
    if key in (ord("d"), ord("D")):
        with scene_lock:
            already = scene_status == "analyzing"
        if not already:
            snap = frame.copy()
            with scene_lock:
                scene_description = ""
                scene_status = "analyzing"
            Thread(target=describe_scene_worker, args=(snap,), daemon=True).start()

    if key in (ord("q"), ord("Q")):
        vstream.stop()
        os._exit(0)
