# Eufy AI Vision

Local computer-vision pipeline for Eufy cameras. Bypasses Eufy's event-only streaming by tunneling P2P video through a local bridge, then runs real-time object tracking, vehicle re-identification, face recognition, and on-demand AI scene description with GPT-5.5 — entirely on your machine, with the video frames never leaving your network unless you press the "describe" key.

Originally built around a Eufy Floodlight Cam S330 (T8423) covering a driveway and a Eufy Video Doorbell (T8200), but the architecture generalizes to any camera supported by the eufy-security-client library.

## What it does

**Driveway camera (`reid_node.py`)**
- Detects cars, trucks, and people in real time using RT-DETR-L on CUDA.
- Tracks each detection across frames with ByteTrack so identities persist between frames.
- Identifies known vehicles by visual signature (HSV color histogram + 16×16 structural fingerprint, 768-dim total, compared via cosine similarity against an enrolled gallery).
- Requires 7 consecutive frame matches with avg similarity > 0.83 before locking an identity — slow but virtually false-positive-proof.
- Press **S** while an unknown vehicle is in frame to enroll it into one of 16 slots.

**Doorbell (`doorbell_node.py`)**
- Detects people only (vehicle Re-ID disabled — wrong tool for a doorbell view).
- For each detected person, finds the largest face with MTCNN and computes a 512-dim FaceNet embedding (InceptionResnetV1, pretrained on VGGFace2) on the GPU.
- Identifies enrolled household members / frequent visitors via cosine similarity against a face gallery.
- Press **S** when a clear frontal face is in frame to enroll it.

**Both pipelines**
- Press **D** to send the current frame to OpenAI GPT-5.5 (Responses API, vision input) and get a 3-5 sentence natural-language description of the scene. Doorbell uses a delivery/visitor-tuned prompt; driveway uses a surveillance-tuned prompt.
- The call runs in a background thread so video and tracking keep running while the model thinks. Result is overlaid on the bottom of the video and logged to stdout.

## Architecture

```
┌──────────────────┐
│   Eufy camera    │ ─── P2P ───┐
└──────────────────┘            │
                                ▼
                  ┌────────────────────────────┐
                  │   eufy-security-server     │  Node.js, ws://localhost:3000
                  │   (bropat's WebSocket      │  Auths with Eufy cloud, opens
                  │    bridge, npm install)    │  P2P sessions to your cameras.
                  └────────────────────────────┘
                                │
                                │  WebSocket (JSON commands + base64 video chunks)
                                ▼
                  ┌────────────────────────────┐
                  │   eufy_bridge*.py          │  Python (asyncio + websockets)
                  │   - sends start_livestream │
                  │   - receives H.264 chunks  │
                  │   - pipes to ffmpeg stdin  │
                  └────────────────────────────┘
                                │
                                │  ffmpeg -c copy -f mpegts udp://127.0.0.1:PORT
                                ▼
                  ┌────────────────────────────┐
                  │   reid_node.py             │  Python + OpenCV + PyTorch (CUDA)
                  │   or doorbell_node.py      │  - RT-DETR person/vehicle detect
                  │                            │  - ByteTrack tracking
                  │                            │  - signature / face match
                  │                            │  - GPT-5.5 scene narration
                  │                            │  - OpenCV window + HUD
                  └────────────────────────────┘
```

Why this architecture? Eufy battery cameras only stream RTSP during motion events. The S330 (wired floodlight) technically has an RTSP toggle in the app but its onboard RTSP server is unreliable, and the eufy-security-server's `start_rtsp_livestream` command isn't supported for this model anyway. The universally-supported `device.start_livestream` command (P2P) is what we use instead. The bridge re-emits its video into a local UDP MPEG-TS stream so OpenCV can read it through its standard ffmpeg backend — no custom decoder.

## Requirements

**Hardware**
- NVIDIA GPU strongly recommended (RT-DETR + FaceNet on CPU will be slow). Tested on RTX 3080.
- A supported Eufy camera. The S330 (T8423) and the original Video Doorbell battery (T8200) are confirmed; most Eufy cameras in the eufy-security-client compatibility list should work.

**Software**
- Windows 10/11 (tested), or Linux/macOS (untested; commands will differ slightly)
- Python 3.12 — `py -3.12` invocation throughout
- Node.js LTS (≥ 20)
- ffmpeg (Windows: install via `winget install Gyan.FFmpeg`)
- NVIDIA driver supporting CUDA 12.4 or later

## Install

```powershell
# 1. Clone
git clone <your-repo-url>
cd <your-repo>

# 2. eufy-security-server (the WebSocket bridge to Eufy's P2P API)
npm install -g eufy-security-ws

# 3. ffmpeg
winget install Gyan.FFmpeg
# Close and reopen PowerShell. Verify with: ffmpeg -version

# 4. Python deps
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install opencv-python ultralytics scikit-learn python-dotenv openai numpy websockets pillow facenet-pytorch

# 5. Swap CPU PyTorch for the CUDA build
# (facenet-pytorch pulls a CPU torch as a dependency; we replace it)
py -3.12 -m pip uninstall torch torchvision torchaudio -y
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 6. Verify CUDA is alive
py -3.12 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Expected:  True NVIDIA GeForce RTX ...
```

## Configure

**`config.json`** for eufy-security-server (put this in some folder, e.g. `C:\eufy-ws\config.json`):

```json
{
  "username": "your_eufy_account@example.com",
  "password": "your_eufy_password",
  "country": "US"
}
```

On first run, eufy-security-server will prompt for 2FA / CAPTCHA in the terminal — Eufy emails the code. Session is cached after that.

**`.env`** in the project folder — copy `.env.example` and fill in. Required variables:

```
# Device serials — find them in the Eufy app under Device Info,
# or in the eufy-security-server logs as "Connected to station T8...".
EUFY_SERIAL=T8423XXXXXXXXXXXX
EUFY_DOORBELL_SERIAL=T8200XXXXXXXXXXXX

# Local stream URLs the bridges output to and the nodes read from.
# Keep the query string — it sizes ffmpeg's UDP buffer to avoid drops.
CAMERA_URL=udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1
DOORBELL_URL=udp://127.0.0.1:5001?fifo_size=5000000&overrun_nonfatal=1

# Absolute path to ffmpeg.exe.
# On Windows after winget install, it lives somewhere like:
FFMPEG_BIN=C:\Users\YOU\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe

# For [D] key scene description. https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...
```

## Run

All pipelines share one `eufy-security-server`. Bridges and nodes pair up per camera.

**Driveway camera (3 windows):**

```powershell
# Window 1 — shared
cd C:\eufy-ws
eufy-security-server --config config.json
# Wait for: Connected to station T8423...

# Window 2
cd <project>
py -3.12 eufy_bridge.py
# Wait for: [stream] N chunks, KB sent to ffmpeg  (counter incrementing)

# Window 3
cd <project>
py -3.12 reid_node.py
```

**Doorbell (3 windows; shares window 1 if it's already running):**

```powershell
# Window 1 — same as above
eufy-security-server --config config.json

# Window 4
py -3.12 eufy_bridge_doorbell.py

# Window 5
py -3.12 doorbell_node.py
```

**All four pipelines together: 5 windows** — Window 1 + windows 2+3 (camera) + windows 4+5 (doorbell). They're independent; killing one doesn't affect the other.

## Controls

In any OpenCV window:

| Key | Action |
|-----|--------|
| **D** | Describe the current scene with GPT-5.5. Result overlays on screen and prints to terminal with timestamp. |
| **S** | Enroll. Driveway: saves the current unknown-vehicle signature. Doorbell: saves the largest currently-visible face. Tk dialog asks for a name and slot (0-15). |
| **Q** | Quit. |

## Tuning

**Vehicle Re-ID** (top of `reid_node.py`):
- `REQUIRED_MATCHES = 7` — number of frames of evidence before locking identity.
- Entry threshold `> 0.80`, lock threshold `> 0.83` — both on cosine similarity over the 768-dim HSV+structure signature.

**Face Re-ID** (top of `doorbell_node.py`):
- `REQUIRED_FACE_MATCHES = 3` — faster than vehicles because face embeddings are more discriminative.
- `ENTER_THRESH = 0.50`, `LOCK_THRESH = 0.60` — cosine similarity over 512-dim L2-normalized FaceNet embeddings.

If you get false positives (someone tagged as the wrong identity), raise the lock thresholds by 0.05. If known faces / vehicles aren't being recognized, lower the entry thresholds by 0.05 or re-enroll with cleaner samples.

## Known quirks

**Eufy P2P sessions are single-viewer.** If anyone opens the Eufy app and looks at the camera, the bridge gets kicked. The bridge auto-restarts within a few seconds, but expect brief interruptions.

**The doorbell battery drains fast under continuous P2P streaming.** Hours, not days. The T8200 is designed to sleep between motion/press events; we keep it awake. For long sessions, pull the doorbell off the mount and put it on a USB charger.

**H.264 decoder warnings on startup are harmless.** `non-existing PPS 0 referenced` and `decode_slice_header error` scroll for a few seconds while OpenCV's decoder waits for the next keyframe + SPS/PPS. The `dump_extra=freq=keyframe` ffmpeg flag we use ensures these arrive within one GOP, then the warnings stop and frames flow normally. The warnings don't affect tracking accuracy.

**When the pipeline hangs**, kill everything, wait 10 seconds (for Eufy's cloud to release the orphaned P2P session), then restart in order: server → bridge → node. Wait for the previous window to show success before starting the next.

**`pip install` can silently downgrade your CUDA PyTorch to a CPU build.** Any package depending on `torch` (e.g. `facenet-pytorch`) will resolve to the CPU build unless you reinstall CUDA torch after. Verify with `torch.cuda.is_available()` after any large pip operation.

**Eufy reverse-engineered tooling lives in a grey zone.** eufy-security-server (and the underlying eufy-security-client) is a community reverse engineering of Eufy's API. It has worked reliably for years, but Eufy occasionally pushes firmware updates that break compatibility. If your bridge suddenly stops working after a camera firmware update, watch the bropat/eufy-security-ws issue tracker.

## Roadmap

Things that would meaningfully improve the system:

- **Replace the vehicle histogram signature with DINOv2 embeddings.** The current HSV+structure signature works but is the weakest part of the pipeline — two cars of the same color and similar silhouette can have very similar signatures. A DINOv2 ViT embedding would be far more discriminative. ~30 lines of changes to `get_advanced_signature()`.
- **Event log persistence.** Write detection events to SQLite ("Mickey's car arrived at 18:42", "Unknown person at doorbell at 09:15"). Enables a separate dashboard.
- **Webhook / notification on unknown identities.** Fire an alert when an unrecognized person stays at the door for >N seconds, or an unrecognized vehicle enters the driveway.
- **Short-clip scene description** instead of single frame. Send a 3-5 second clip to GPT-5.5 so it can describe motion ("a person walked up, knocked, then turned away") instead of just stills.
- **Tiled multi-camera view** with hotkey switching between full-screen feeds.

## Files

```
reid_node.py                 - Driveway pipeline (cars + people + GPT-5.5)
eufy_bridge.py               - P2P-to-UDP bridge for the camera
doorbell_node.py             - Doorbell pipeline (people + faces + GPT-5.5)
eufy_bridge_doorbell.py      - P2P-to-UDP bridge for the doorbell
.env.example                 - Template for your local .env
.gitignore                   - Excludes secrets and personal data
README.md                    - You are here
```

## Acknowledgments

- [bropat/eufy-security-ws](https://github.com/bropat/eufy-security-ws) and [eufy-security-client](https://github.com/bropat/eufy-security-client) — the entire bridge ecosystem this project rides on.
- [Ultralytics](https://github.com/ultralytics/ultralytics) — RT-DETR + ByteTrack.
- [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN + InceptionResnetV1.
- OpenAI — GPT-5.5 vision model for scene description.

## Disclaimer

This project uses unofficial third-party tooling to access Eufy cameras. It does not use, modify, or distribute any code from Anker/Eufy. Use of the eufy-security-client library may technically violate Eufy's Terms of Service; the legal status of reverse-engineered camera access in your jurisdiction is your responsibility to understand. All video processing happens locally; no frames leave your machine except when you press `D` to invoke the OpenAI vision API, at which point the current frame is uploaded for description.