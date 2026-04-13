# Enterprise Vehicle Re-Identification Node

![Vehicle Tracking System Demo](demo.jpg)

## Overview
A real-time, multi-threaded computer vision node designed to track, identify, and log vehicles entering a residential driveway. Built using Python, OpenCV, and the Ultralytics RT-DETR model, this system acts as an "AI Concierge," maintaining a 3D memory database of known vehicles while actively filtering out false positives.

It is specifically optimized to handle the network quirks and RTSP dropouts associated with battery-powered Eufy security cameras.

## Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (cv2)
* **Deep Learning / AI:** Ultralytics (RT-DETR-L), ByteTrack
* **Mathematics / Matrix Operations:** NumPy, scikit-learn (Cosine Similarity)
* **Architecture:** Multi-threading, Asynchronous Video Streaming

## Key Features
* **16-Slot Identity Database:** Stores the mathematical signatures (color histograms and structural features) of up to 16 known vehicles.
* **Rolling Average Validation:** Implements a 7-frame validation protocol to calculate a rolling average confidence score, completely eliminating false positives from similar-looking vehicles passing by.
* **Battery-Camera Standby Mode:** Features a custom VideoStream thread that gracefully handles 404 connection drops when the battery-powered camera enters sleep mode, rendering a Standby UI instead of freezing the OS.
* **Gagged FFmpeg Logging:** Overrides internal C++ FFmpeg logging to prevent terminal spam during network disconnects.

## Under the Hood: The Re-ID Pipeline
1. **Detection & Tracking:** The system uses the RT-DETR-L transformer model to detect vehicles and ByteTrack to assign consistent frame-to-frame IDs.
2. **Feature Extraction:** When an unknown vehicle is detected, the script crops the bounding box and generates a highly specific 1D mathematical signature. This signature combines a 3D HSV color histogram with structural grayscale features.
3. **Similarity Scoring:** The live signature is compared against the 16-slot `.npy` database using Cosine Similarity.
4. **Validation:** To eliminate false positives from similar-looking vehicles (e.g., a neighbor's car), the system stores the similarity scores in a dictionary. It requires an average confidence score of >83% across 7 consecutive frames before locking the identity and turning the bounding box Gold.

## Prerequisites
To run this script, you will need a `.env` file in the root directory containing your camera's RTSP URL:
`CAMERA_URL="rtsp://your_username:your_password@your_ip_address/live0"`

### Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt