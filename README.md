# 🐘 Elephant Detector

> **An AI-powered real-time elephant detection and early-warning system.**

Elephant Detector is an AI/computer-vision project designed to detect elephants from images and live camera feeds using **YOLO object detection**.

The initial goal is simple:

**Camera → AI Detection → Confidence Filter → Persistence Verification → Cooldown Alert**

The project is being developed incrementally, starting with a software-only prototype and eventually evolving toward an edge-AI system capable of operating in real-world environments.

---

## 🚨 Why This Project?

Human-elephant conflict is a serious problem in many regions where forests and human settlements overlap.

A traditional camera can capture an elephant, but it cannot automatically understand what it sees.

This project adds an AI layer:

```text
📷 Camera
   ↓
🤖 Computer Vision (YOLO)
   ↓
🐘 Elephant Detected
   ↓
📊 Confidence & Persistence Filter
   ↓
🚨 Warning / Alert (With Cooldown)
```

The long-term vision is to create an intelligent monitoring system that can detect elephants early and provide timely warnings to people in potentially affected areas.

---

# 🎯 Current Status: Phase 2 Prototype

The project has completed **Phase 2 (Real-Time Webcam Detection System)**.

### Phase 2 Architecture Pipeline

```text
                  📷 WEBCAM
                      ↓
             🎞️ OpenCV Capture
                      ↓
          🤖 YOLO Model (yolo26n.pt)
                      ↓
              OBJECT DETECTION
                      ↓
               🐘 ELEPHANT?
                  ↙       ↘
         NO / OTHER        YES
             ↓              ↓
      Display Box Only  Confidence Check (>= 70%)
      (Person/Car/etc.)     ↓
                        Persistence Check (5 consecutive frames)
                            ↓
                        Confirmed!
                            ↓
                    🚨 ELEPHANT ALERT
                            ↓
                       30s Cooldown
                            ↓
                      Keep Monitoring
```

---

# 🧠 Technology Stack

## Core AI & Libraries

* **Python 3.10+**
* **Ultralytics YOLO** (Object detection inference)
* **OpenCV (`opencv-python`)** (Video stream capture, HUD rendering, and UI display)
* **PyTorch** (Deep learning backend)

---

# 📁 Project Structure

```text
Elephant_detector/
│
├── elephant_camera.py   # Real-time webcam detector with persistence, cooldown, and HUD
├── detect.py            # Static image detection script
├── camera.py            # Basic OpenCV webcam test script
├── test_phase2.py       # Automated verification test suite for Phase 2
├── elephant.jpg         # Sample test image
├── yolo26n.pt           # Pretrained YOLO model weights
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation and roadmap
```

---

# ⚙️ Getting Started

## 1. Clone or Open the Repository

```powershell
cd Projects\Elephant_detector
```

## 2. Set Up Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

Install the verified dependencies using `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

---

# 🚀 Running the Detector

## 1. Real-Time Camera Detection (Phase 2)

Launch the real-time webcam elephant detector:

```powershell
python elephant_camera.py
```

### Controls & Features

* **Exit**: Press `Q` or `q` at any time to release the camera and close the window safely.
* **Bounding Boxes**:
  * 🔴 **Red Box**: Confirmed elephant detection meeting confidence threshold (`ELEPHANT: 96%`).
  * 🟡 **Yellow Box**: Elephant candidate below the confidence threshold (`elephant (low conf): 54%`).
  * 🔵 **Cyan Box**: Other detected objects (e.g. `person: 82%`, `car: 75%`) displayed without triggering alerts.
* **On-Screen HUD**:
  * `STATUS: MONITORING` — Normal state (green badge).
  * `STATUS: POSSIBLE ELEPHANT (X/5)` — Elephant detected, verifying persistence (amber badge).
  * `STATUS: ELEPHANT CONFIRMED` — Elephant confirmed after 5 consecutive frames (red badge).
  * `Cooldown Active: Xs remaining` — Shows remaining cooldown time while monitoring continues.
  * `FPS: XX.X` — Real-time frame processing rate.
* **Alerts**:
  * 🚨 Console alert logged with timestamp and confidence score.
  * Prominent red visual alert banner rendered on the video window.

---

## 2. Static Image Detection

To run detection on a single image file:

```powershell
python detect.py
```

---

## 3. Run Automated Logic Tests

To verify detection, persistence counter, counter reset, alert triggering, and cooldown logic:

```powershell
python test_phase2.py
```

---

# ⚙️ Configuration

All key parameters are easily configurable at the top of [`elephant_camera.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/elephant_camera.py):

```python
# Path to the pretrained YOLO model weights
MODEL_PATH = "yolo26n.pt"

# Minimum confidence required to accept an elephant detection (70%)
CONFIDENCE_THRESHOLD = 0.70

# Number of consecutive frames an elephant must be detected before alert
REQUIRED_DETECTIONS = 5

# Time in seconds to wait before allowing another alert (prevents alert spam)
ALERT_COOLDOWN_SECONDS = 30

# Duration in seconds to display the visual emergency banner
ALERT_BANNER_DURATION_SECONDS = 4.0

# Target class name to monitor
TARGET_CLASS = "elephant"

# Default webcam index (0 is standard default camera)
CAMERA_INDEX = 0
```

---

# 🔬 Phase 2 Key Mechanisms

### 1. Confidence Filtering
Prevents low-confidence noise from registering as an elephant. Only detections with `confidence >= 0.70` (70%) are counted toward an alert.

### 2. Multi-Frame Persistent Detection
Single-frame detection anomalies (glitches, reflections, false positives) are ignored. An alert requires `REQUIRED_DETECTIONS = 5` consecutive frames of confirmed elephant presence.

### 3. Automatic Counter Reset
If an elephant leaves the frame or is no longer detected, the consecutive detection counter immediately resets to `0`, returning the system to `MONITORING`.

### 4. Alert Cooldown
When an elephant is confirmed and an alert is dispatched, an `ALERT_COOLDOWN_SECONDS = 30` cooldown is initiated. During cooldown:
* The video stream continues processing smoothly.
* Visual indicators remain active.
* Terminal alerts are not spammed every frame.
* A new alert is only triggered if an elephant remains or reappears after the cooldown expires.

---

# ⚠️ Current Limitations (Phase 2)

* **Pretrained General YOLO Model**: The current system uses `yolo26n.pt` trained on general COCO classes. While it identifies elephants well in clear standard photos, it may struggle with occluded elephants, herd formations, night footage, or distant animals in dense foliage.
* **Lighting Dependency**: Standard RGB webcams are sensitive to poor illumination and night conditions.
* **Local Alerts Only**: Alerts are currently local (console + OpenCV HUD banner). Remote alerting (SMS/WhatsApp/Cloud/Siren) is planned for subsequent phases.

These limitations are expected and will be directly solved in **Phase 3 (Custom Dataset & Fine-Tuning)** and **Phase 4 (Tracking & Intelligent Early Warning)**.

---

# 🧪 Development Roadmap

### Phase 1 — Basic Detection
* [x] Python environment setup
* [x] Install YOLO & OpenCV
* [x] Load pretrained model (`yolo26n.pt`)
* [x] Static image detection (`detect.py`)
* [x] Understand bounding boxes & confidence scores

### Phase 2 — Real-Time Detection (Current)
* [x] Webcam input integration via OpenCV
* [x] Real-time YOLO inference pipeline
* [x] Target class filtering (`TARGET_CLASS = "elephant"`)
* [x] Configurable confidence threshold (`CONFIDENCE_THRESHOLD = 0.70`)
* [x] Persistent multi-frame detection counter (`REQUIRED_DETECTIONS = 5`)
* [x] Detection counter reset logic
* [x] Alert cooldown rate limiter (`ALERT_COOLDOWN_SECONDS = 30`)
* [x] Heads-Up Display (HUD) with real-time state, persistence progress, and FPS
* [x] Non-blocking visual alert banner and timestamped terminal alerts
* [x] Clean shutdown handling with `Q` key

### Phase 3 — Custom Elephant Model (Next)
* [ ] Collect specialized elephant dataset (wildlife, dense foliage, varying angles)
* [ ] Annotate images (bounding boxes & posture)
* [ ] Create train/validation/test splits
* [ ] Fine-tune YOLO on custom dataset
* [ ] Benchmark Precision, Recall, and mAP
* [ ] Reduce false positives in wilderness environments

### Phase 4 — Intelligent Tracking & Movement
* [ ] Multi-object tracking (ByteTrack / BoT-SORT)
* [ ] Track individual elephants across frames
* [ ] Estimate direction of movement (approaching vs departing)
* [ ] Detect herd count and clustering
* [ ] Risk scoring engine

### Phase 5 — Early Warning System & Remote Notifications
* [ ] SMS / WhatsApp / Telegram alert dispatcher
* [ ] Web monitoring dashboard
* [ ] Geofencing and localized warning zones
* [ ] Sound / Siren alarm trigger

### Phase 6 — Edge AI Deployment
* [ ] Port pipeline to edge hardware (NVIDIA Jetson / Raspberry Pi)
* [ ] IR / Thermal camera sensor fusion for nighttime detection
* [ ] Solar & battery power management
* [ ] Low-power standby and wake-on-motion

---

## ⭐ Contributing & License

This project is built for **wildlife conservation, human-wildlife conflict mitigation, and community safety**.

Contributions and suggestions are welcome.
