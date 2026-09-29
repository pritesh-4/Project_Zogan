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

# 🎯 Current Status: Phase 3 Complete (Custom Elephant Model Integrated)

The project has achieved full integration of our fine-tuned custom YOLO elephant detection model (`elephant_v1`) into the real-time webcam detection system (`elephant_camera.py`), replacing the generic COCO model while preserving all Phase 2 persistence, cooldown, and HUD logic.

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
├── ai/
│   ├── train.py              # Fine-tuning transfer learning script
│   ├── evaluate.py           # Model validation, mAP reporting, and test prediction generator
│   ├── predict_custom.py     # Single-image inference using the custom elephant model
│   ├── compare_models.py     # Side-by-side comparison between baseline and custom models
│   ├── validate_dataset.py   # Dataset structure and annotation validation utility
│   └── visualize_dataset.py  # Visual inspection tool for YOLO labels and bounding boxes
│
├── datasets/
│   └── elephant/
│       ├── images/           # Images organized by split (train: 315, val: 68, test: 73)
│       ├── labels/           # Matching YOLO label text files (train: 315, val: 68, test: 73)
│       ├── data.yaml         # Dataset configuration file for Ultralytics YOLO
│       ├── metadata.json     # Dataset manifest and original source tracking
│       └── README.md         # Comprehensive dataset & YOLO annotation guide
│
├── models/
│   ├── README.md             # Model changelog, deployment specifications, and version guide
│   └── elephant_v1/
│       ├── best.pt           # Deployed fine-tuned custom elephant model weights
│       └── README.md         # Experiment notes
│
├── runs/
│   └── detect/
│       └── elephant_v1/      # Full training run artifacts, plots, and MODEL_REPORT.md
│
├── elephant_camera.py        # Real-time webcam detector with custom model, persistence, HUD
├── detect.py                 # Static image detection script
├── camera.py                 # Basic OpenCV webcam test script
├── test_phase2.py            # Automated verification test suite for Phase 2
├── test_phase3_1.py          # Automated verification test suite for Phase 3.1
├── test_phase3_2.py          # Automated verification test suite for Phase 3.2
├── test_phase3_4.py          # Automated verification test suite for Phase 3.4
├── elephant.jpg              # Sample test image
├── yolo26n.pt                # Pretrained base YOLO model weights
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation and roadmap
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

---

## Running the Custom Model

When custom-model mode is enabled, Project Zogan's live detector loads our validated, specialized model:
`runs/detect/elephant_v1/weights/best.pt` (or `models/elephant_v1/best.pt`).

### A. Real-Time Detection with Custom Model (Live Detector)
By default, [`elephant_camera.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/elephant_camera.py) now automatically loads our custom fine-tuned model (`elephant_v1`):

```powershell
python elephant_camera.py
```

At startup, the detector displays:
```text
============================================================
🐘 PROJECT ZOGAN — ELEPHANT DETECTION SYSTEM
Model: elephant_v1
Weights: models/elephant_v1/best.pt
Target Class: elephant
Confidence Threshold: 70%
Required Detections: 5
Alert Cooldown: 30s
============================================================
```

### B. Switching Back to Pretrained Model
To run side-by-side A/B testing or switch back to the generic COCO pretrained model (`yolo26n.pt`):

```powershell
python elephant_camera.py --pretrained
```
Alternatively, set `USE_CUSTOM_MODEL = False` in [`elephant_camera.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/elephant_camera.py).

### C. Single Image Custom Prediction
Run detection on any test image using our custom model directly:

```powershell
python ai/predict_custom.py elephant.jpg
```

### D. Side-by-Side Model Comparison
Compare detections, confidences, and inference latencies between baseline `yolo26n.pt` and custom `elephant_v1` on the exact same image:

```powershell
python ai/compare_models.py elephant.jpg
```

### E. Model Training & Evaluation Pipeline
To retrain or re-evaluate the custom model:

```powershell
# Validate dataset integrity
python ai/validate_dataset.py

# Train custom model (transfer learning from yolo26n.pt)
python ai/train.py --epochs 15 --imgsz 416 --batch 16

# Evaluate on validation and unseen test sets
python ai/evaluate.py
```

---

## 4. Run Automated Test Suites

To verify Phase 2 real-time detection, persistence, cooldown, and HUD rendering:

```powershell
python test_phase2.py
```

To verify Phase 3.1 dataset infrastructure, YAML configuration, and validator error handling:

```powershell
python test_phase3_1.py
```

To verify Phase 3.2 custom dataset integrity, counts, zero leakage, decoding, and labels:

```powershell
python test_phase3_2.py
```

To verify Phase 3.4 custom model integration, model switching, class mapping, and end-to-end pipeline:

```powershell
python test_phase3_4.py
```

---

# ⚙️ Configuration

All key parameters are easily configurable at the top of [`elephant_camera.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/elephant_camera.py):

```python
# Model selection configuration
USE_CUSTOM_MODEL = True
CUSTOM_MODEL_PATH = "models/elephant_v1/best.pt"
FALLBACK_MODEL_PATH = "runs/detect/elephant_v1/weights/best.pt"
PRETRAINED_MODEL_PATH = "yolo26n.pt"

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

# ⚠️ Current Limitations (Phase 3.4)

* **Pretrained vs Custom Domain**: `elephant_v1` was trained predominantly on African Savannah elephants. Domain shift on Asian forest elephants, extreme night footage, or adverse weather may require future synthetic or domain-specific dataset expansion.
* **Lighting Dependency**: Standard RGB webcams are sensitive to poor illumination and night conditions.
* **Local Alerts Only**: Alerts are currently local (console + OpenCV HUD banner). Remote alerting (SMS/WhatsApp/Cloud/Siren) is planned for subsequent phases.

These limitations are tracked for **Phase 4 (Tracking & Intelligent Early Warning)** and future dataset expansion.

---

# 🧪 Development Roadmap

### Phase 1 — Basic Detection
* [x] Python environment setup
* [x] Install YOLO & OpenCV
* [x] Load pretrained model (`yolo26n.pt`)
* [x] Static image detection (`detect.py`)
* [x] Understand bounding boxes & confidence scores

### Phase 2 — Real-Time Detection
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

### Phase 3 — Custom Elephant Model
* [x] Dataset preparation
* [x] Dataset validation
* [x] Custom dataset
* [x] Annotation
* [x] Model training
* [x] Model evaluation
* [x] Custom model integration
* [ ] Model improvement (Asian elephants, infrared/night vision, adverse weather)

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
