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

# 🎯 Current Status: Phase 5 Complete (GPS, Geofencing & Basic Risk Assessment)

Project Zogan has implemented a software-based geographic intelligence layer, including **simulated camera coordinates**, **multi-tier geofenced zones (Village, Buffer, Forest)**, **Haversine geodesic distance calculation**, **approach trend detection**, and a **rule-based early-warning risk engine (LOW, MEDIUM, HIGH, CRITICAL)** layered seamlessly on top of our fine-tuned custom elephant detector (`elephant_v1`) and ByteTrack multi-object tracker.

### Phase 5 Architecture Pipeline

```text
                  📷 CAMERA / VIDEO REPLAY
                             ↓
             🤖 CUSTOM YOLO MODEL (elephant_v1)
                             ↓
            🐘 ELEPHANT DETECTION (conf >= 0.70)
                             ↓
                   🎯 BYTETRACK TRACKER
                             ↓
                  🏷️ TRACK ID (#1, #2...)
                             ↓
             📈 POSITION HISTORY & IMAGE MOVEMENT
                             ↓
            🗺️ GEOFENCING & ZONE CLASSIFICATION
               (Village / Buffer / Forest / Outside)
                             ↓
                 🧭 APPROACH TREND TRACKING
               (APPROACHING / RECEDING / STABLE)
                             ↓
                 ⚖️ RULE-BASED RISK ENGINE
               (LOW / MEDIUM / HIGH / CRITICAL)
                             ↓
         🖥️ HUD WITH GEO CONTEXT & RISK-AWARE ALERTS
```

---

# 🧠 Technology Stack

## Core AI & Libraries

* **Python 3.10+**
* **Ultralytics YOLO** (Object detection inference & ByteTrack tracking)
* **OpenCV (`opencv-python`)** (Video stream capture, HUD rendering, and UI display)
* **PyTorch** (Deep learning backend)
* **Lap (`lap`)** (Linear assignment solver for ByteTrack bipartite matching)
* **Math (Standard Library)** (Haversine geodesic distance and spherical trigonometry)

---

# 📁 Project Structure

```text
Elephant_detector/
│
├── config.py                 # Central configuration for simulated GPS, zones, and risk thresholds
│
├── ai/
│   ├── geofence.py           # Geofencing, Haversine distance, and zone classification (Phase 5)
│   ├── risk_engine.py        # Rule-based early-warning risk scoring engine (Phase 5)
│   ├── simulate_risk.py      # Trajectory and approach risk simulation utility (Phase 5)
│   ├── tracking.py           # Object tracking and movement analysis module (Phase 4)
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
├── elephant_camera.py        # Real-time webcam & video tracker with custom model, geofencing, HUD
├── detect.py                 # Static image detection script
├── camera.py                 # Basic OpenCV webcam test script
├── test_phase2.py            # Automated verification test suite for Phase 2
├── test_phase3_1.py          # Automated verification test suite for Phase 3.1
├── test_phase3_2.py          # Automated verification test suite for Phase 3.2
├── test_phase3_4.py          # Automated verification test suite for Phase 3.4
├── test_phase4.py            # Automated verification test suite for Phase 4
├── test_phase5.py            # Automated verification test suite for Phase 5
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

To verify Phase 4 object tracking, track IDs, center points, bounded history, image-space movement, anti-jitter threshold, multi-elephant tracking, and lost-track expiration:

```powershell
python test_phase4.py
```

To verify Phase 5 geofencing, Haversine geodesic distance, zone classification, approach/recede trends, and rule-based risk evaluation:

```powershell
python test_phase5.py
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

---

# 🐘 Phase 4 — Object Tracking & Movement Analysis

Phase 4 introduces multi-object elephant tracking across video frames using **Ultralytics ByteTrack**, paired with 2D image-space movement estimation, bounded position history, anti-jitter filtering, and real-time HUD metrics.

### 1. Why Tracking Is Needed
Previous phases treated each video frame independently:
- Frame 1 → Elephant detected
- Frame 2 → Elephant detected
- Frame 3 → Elephant detected

The detector had no concept of temporal identity or persistence over time. Phase 4 introduces tracking so the system recognizes:
- Frame 1 → Elephant ID #1
- Frame 2 → Elephant ID #1
- Frame 3 → Elephant ID #1
- Frame 4 → Elephant ID #1

**"We are following the same elephant."**

This allows us to maintain position history, suppress bounding box jitter, and analyze whether an elephant is stationary or moving across the field of view.

### 2. Pipeline Architecture
```text
      📷 CAMERA / VIDEO REPLAY
                 ↓
     🤖 CUSTOM YOLO MODEL (elephant_v1)
                 ↓
    🐘 ELEPHANT DETECTION (conf >= 0.70)
                 ↓
       🎯 BYTETRACK TRACKER
                 ↓
      🏷️ TEMPORARY TRACK ID (#1, #2...)
                 ↓
     📈 POSITION HISTORY (MAX_HISTORY = 20)
                 ↓
 🧭 DIRECTION ESTIMATION (RIGHT/LEFT/UP/DOWN/STATIONARY)
                 ↓
  🖥️ HUD & ENHANCED LOCAL ALERTS (WITH COOLDOWN)
```

### 3. Key Tracking Capabilities
* **Temporary Track IDs**: Each detected elephant is assigned a lightweight session ID (e.g. `ELEPHANT #1`, `ELEPHANT #2`) using ByteTrack's bipartite Hungarian matching.
* **Center-Point Calculation**: Every bounding box is mapped to its center coordinate:
  $$\text{center}_x = \frac{x_1 + x_2}{2}, \quad \text{center}_y = \frac{y_1 + y_2}{2}$$
* **Bounded Position History**: Maintains up to `MAX_HISTORY = 20` points per tracked elephant, preventing memory bloat.
* **Image-Space Movement Direction**: Computes dominant movement vector:
  * `RIGHT`: $\Delta x > 0$ and $|\Delta x| \ge |\Delta y|$
  * `LEFT`: $\Delta x < 0$ and $|\Delta x| \ge |\Delta y|$
  * `UP`: $\Delta y < 0$ and $|\Delta y| > |\Delta x|$ (in image coordinates, $y$ decreases upwards)
  * `DOWN`: $\Delta y > 0$ and $|\Delta y| > |\Delta x|$
  * `STATIONARY`: Displacement $< \text{MOVEMENT\_THRESHOLD\_PIXELS}$ (default: 10 px)
  * `UNKNOWN`: New track with insufficient frame history ($< 2$ frames)
* **Anti-Jitter Suppression**: Bounding boxes naturally fluctuate by several pixels frame-to-frame. The configurable movement threshold (`MOVEMENT_THRESHOLD_PIXELS = 10.0`) and multi-frame window smoothing ensure minor detector noise is classified as `STATIONARY`.
* **Multiple Elephant Tracking**: Simultaneously maintains separate histories, IDs, and movement vectors for multiple elephants in frame.
* **Lost-Track Tolerance**: Tolerates temporary occlusions or dropped detections for up to 30 frames (`track_buffer = 30`) before cleanly evicting expired tracks.
* **Video File Replay Option**: Process recorded video files for repeatable verification:
  ```powershell
  python elephant_camera.py --source elephant_video.mp4
  ```

---

# ⚠️ Important Limitations (Phase 4)

Clearly documented constraints and boundary conditions:

1. **Direction is image-space direction only**: Movement labels (`RIGHT`, `LEFT`, `UP`, `DOWN`) reflect movement within the camera's 2D sensor frame. They do **NOT** indicate geographic or compass headings (`NORTH`, `SOUTH`, `EAST`, `WEST`).
2. **The camera is assumed stationary**: Image-space movement is valid only when the camera itself does not move. Camera pan, tilt, shake, or vehicle motion will produce false motion vectors. Camera ego-motion compensation is not yet implemented.
3. **Tracking IDs are temporary session IDs**: IDs identify tracked objects during the current session only. They do **NOT** represent permanent biological identities of individual elephants.
4. **Movement speed is not yet real-world speed**: Movement displacement is measured in pixels per frame, not meters per second.
5. **Pixel movement does not equal physical distance**: Because of perspective distortion, an elephant moving 20 pixels in the background has traveled a much greater physical distance than one moving 20 pixels in the foreground.
6. **No geographic location is known yet**: The system operates without GPS coordinates, boundary maps, or geofencing zones.
7. **No risk score exists yet**: Movement direction is not connected to danger or threat levels (e.g. moving "RIGHT" does not imply heading toward a human village).
8. **The system does not predict elephant behavior**: No behavioral intent, aggression modeling, or herd intention is inferred.

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

### Phase 4 — Intelligent Tracking
* [x] Object tracking (ByteTrack)
* [x] Track IDs (Temporary session IDs: ELEPHANT #1, #2...)
* [x] Position history (Bounded to 20 centers)
* [x] Movement estimation (Image-space: RIGHT, LEFT, UP, DOWN, STATIONARY)
* [x] Multiple elephant tracking
* [x] Basic movement HUD & Enhanced alert info
* [ ] Camera motion compensation
* [ ] Geographic movement
* [ ] Risk assessment

---

# 🗺️ Phase 5 — GPS, Geofencing & Basic Risk Assessment

Phase 5 introduces a software-only spatial intelligence layer combining simulated camera GPS coordinates, multi-tier geofenced zones, geodesic Haversine distance calculations, approach trend tracking, and a rule-based early-warning risk scoring engine.

### 1. Why Geographic Context Matters
In Phase 4, the system could detect that an elephant moved "RIGHT" or "LEFT". However:
- "RIGHT" does not mean "EAST".
- "RIGHT" does not mean "TOWARD THE VILLAGE".
- A camera facing North sees "RIGHT" as East; a camera facing South sees "RIGHT" as West.

Without geographic anchoring, image-space movement cannot assess whether an animal is approaching human settlements. Phase 5 adds this spatial grounding in software.

### 2. Geofenced Zones Model
The system defines concentric and polygonal spatial zones configured relative to the protected human community:
1. **Protected / Village Settlement Zone** (High priority core protection area, e.g. 300m radius around settlement).
2. **Buffer / Warning Zone** (Intermediate corridor between wilderness and village, e.g. 800m perimeter).
3. **Forest / Monitoring Zone** (Natural habitat monitoring perimeter, e.g. 2000m perimeter).
4. **Outside** (Beyond monitored territory).

### 3. Geodesic Calculations & Distance
All spatial distances are computed using the **Haversine formula** on the WGS84 sphere, returning real-world distances in meters:
```python
distance = calculate_distance(lat1, lon1, lat2, lon2)
```
- Example: Camera at `(20.1234, 85.1234)`, Target at `(20.1240, 85.1240)` $\longrightarrow$ **91.52 meters (~90m)**.

### 4. Approach Trend Stability
Rather than reacting to single-frame noise, the `MovementTrendTracker` evaluates the distance delta over a multi-observation window:
- $\Delta d \le -15.0\text{m} \longrightarrow$ **`APPROACHING`** (Elephant is moving closer to protected zone)
- $\Delta d \ge +15.0\text{m} \longrightarrow$ **`RECEDING`** (Elephant is moving away)
- $|\Delta d| < 15.0\text{m} \longrightarrow$ **`STABLE`** (Elephant is stationary or milling within noise margin)
- History $< 2$ points $\longrightarrow$ **`UNKNOWN`**

### 5. Rule-Based Risk Engine
The system evaluates a transparent numerical score ($0–100$) based on 5 objective factors:
1. **Zone Severity** (Max 40 pts): Village (40), Buffer (20), Forest (5), Outside (0)
2. **Proximity to Protected Zone** (Max 25 pts): 0m (25), $\le 200$m (20), $\le 500$m (15), $\le 1000$m (5)
3. **Approach Trend** (Max 20 pts): `APPROACHING` (+20), `STABLE`/`UNKNOWN` (+5), `RECEDING` (0)
4. **Group Size / Herd Count** (Max 10 pts): $\ge 4$ elephants (10), 2–3 elephants (5), 1 elephant (2)
5. **Detection Confidence** (Max 10 pts): $\ge 90\%$ (10), $\ge 80\%$ (7), $\ge 70\%$ (5)

Scores map to 4 actionable levels:
- **`0 – 24` $\longrightarrow$ `LOW`**: Far in forest, stable/receding. Continue normal monitoring.
- **`25 – 49` $\longrightarrow$ `MEDIUM`**: In forest approaching, or in buffer zone stable. Elevated visual monitoring.
- **`50 – 74` $\longrightarrow$ `HIGH`**: In buffer corridor approaching settlement. Triggers local early-warning alert!
- **`75 – 100` $\longrightarrow$ `CRITICAL`**: Inside or directly breaching protected village perimeter. Triggers high-priority emergency alarm!

### 6. Trajectory Simulation Script
Run the automated trajectory simulation to observe the risk engine evaluate an approaching sequence:
```powershell
python ai/simulate_risk.py                    # Approach simulation (1000m down to 0m)
python ai/simulate_risk.py --scenario recede  # Receding simulation (150m out to 1200m)
python ai/simulate_risk.py --group-size 4     # Herd approach simulation
```

---

# ⚠️ Important Limitations & Safety Disclaimers (Phase 5)

Clearly documented constraints and ethical boundaries:

1. **Experimental Prototype Only**: This is an experimental decision-support prototype. It is NOT an official wildlife-management system.
2. **Not a Validated Biological Model**: The risk score is rule-based and not a scientifically validated animal-behavior model. An elephant is NOT inherently dangerous simply because it exists; risk is defined purely by spatial proximity to human agriculture/settlements.
3. **No Intent or Behavioral Prediction**: The system cannot predict elephant mood, intent, or charging behavior.
4. **Camera GPS $\neq$ Elephant GPS**: Camera coordinates are known. An RGB camera cannot determine real-world GPS coordinates without rangefinding/depth sensors. In this phase, elephant coordinates are **simulated** in software.
5. **No Physical GPS Hardware**: No physical GPS modules (NEO-6M, u-blox) are integrated yet.
6. **No Remote Alerts Yet**: Alerts remain local (OpenCV HUD banner and terminal output). Remote notification (SMS/WhatsApp/Telegram) is planned for Phase 6.
7. **Stationary Camera Assumption**: Camera orientation and position are assumed static; camera motion compensation is not yet supported.

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

### Phase 4 — Intelligent Tracking
* [x] Object tracking (ByteTrack)
* [x] Track IDs (Temporary session IDs: ELEPHANT #1, #2...)
* [x] Position history (Bounded to 20 centers)
* [x] Movement estimation (Image-space: RIGHT, LEFT, UP, DOWN, STATIONARY)
* [x] Multiple elephant tracking
* [x] Basic movement HUD & Enhanced alert info
* [ ] Camera motion compensation
* [ ] Geographic movement
* [ ] Risk assessment

### Phase 5 — GPS, Geofencing & Basic Risk Assessment
* [x] Geographic utility module (`ai/geofence.py`)
* [x] Simulated camera coordinates (`config.py`)
* [x] Geofence support (Village, Buffer, Forest)
* [x] Distance calculation (Haversine formula in meters)
* [x] Zone classification (Village > Buffer > Forest > Outside)
* [x] Basic approach trend (APPROACHING / RECEDING / STABLE)
* [x] Rule-based risk engine (LOW, MEDIUM, HIGH, CRITICAL)
* [x] Simulation mode & test script (`ai/simulate_risk.py`)
* [x] Risk-aware local alerts & HUD context
* [ ] Real GPS hardware
* [ ] Camera heading calibration
* [ ] Real-world elephant coordinate estimation
* [ ] Remote notification system

### Phase 6 — Remote Alerting & Event Logging
* [ ] SMS / WhatsApp / Telegram alert dispatcher
* [ ] Web monitoring dashboard
* [ ] Geofencing and localized warning zones
* [ ] Sound / Siren alarm trigger

### Phase 7 — Edge AI Deployment
* [ ] Port pipeline to edge hardware (NVIDIA Jetson / Raspberry Pi)
* [ ] IR / Thermal camera sensor fusion for nighttime detection
* [ ] Solar & battery power management
* [ ] Low-power standby and wake-on-motion

---

## 🛡️ Continuous Integration & Quality Gate (GitHub Actions)

Project Zogan uses an automated GitHub Actions CI/CD pipeline defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) to guard against regressions, syntax errors, import mismatches, and broken test suites.

```text
CODE CHANGE / PULL REQUEST
            ↓
  GitHub Actions CI (Ubuntu)
            ↓
  1. Checkout & Setup Python 3.11 (with pip cache)
            ↓
  2. Install headless system libs (libgl1, libglib2.0-0)
            ↓
  3. Install dependencies (requirements.txt & requirements-dev.txt)
            ↓
  4. Python syntax compilation (python -m compileall)
            ↓
  5. Core imports verification (torch, cv2, ultralytics, lap)
            ↓
  6. Code quality & lint gate (ruff check .)
            ↓
  7. Code formatting gate (ruff format --check .)
            ↓
  8. Tracked secret detection (scan for .env, .pem, .key)
            ↓
  9. Dataset & YAML configuration integrity (validate_dataset.py)
            ↓
 10. Automated Test Suites (Phases 2, 3.1, 3.2, 3.4, 4, 5 & Risk simulation)
            ↓
  ✅ PASS → Allowed to merge into main
  ❌ FAIL → Merge blocked
```

### CI Pipeline Features
* **Automatic Triggers**: Runs on all pull requests targeting `main` and pushes to `main`.
* **Fast Failure**: Fails immediately on any syntax error, missing dependency, failing test, or lint error.
* **Concurrency Control**: Automatically cancels outdated runs when a newer commit is pushed to the same pull request (`cancel-in-progress: true`).
* **Hardware-Safe**: GitHub-hosted runners execute software-only and headless tests. Hardware-dependent operations (live webcam feeds, physical sensors) are kept out of CI.
* **No In-CI Model Training**: CI evaluates existing models and verifies integrity without running compute-heavy training loops.

---

## 🧪 Local CI Checks

Before pushing changes or creating a Pull Request, developers should run the exact same checks locally:

```bash
# 1. Validate Python syntax across all project files
python -m compileall -q -x "venv|\.venv" .

# 2. Check imports and core environment
python -c "import torch, cv2, ultralytics, yaml, lap; print('Core dependencies OK!')"

# 3. Code quality lint check
ruff check .

# 4. Code formatting check (check-only mode)
ruff format --check .

# (Optional) Auto-format files locally
ruff format .
ruff check --fix .

# 5. Dataset and configuration validation
python ai/validate_dataset.py

# 6. Run automated test suites (Headless & Safe)
python test_phase2.py
python test_phase3_1.py
python test_phase3_2.py
python test_phase3_4.py
python test_phase4.py
python test_phase5.py
python ai/simulate_risk.py
```

---

## 🔒 Main Branch Protection Policy

To guarantee repository integrity, direct pushes to `main` should be disabled, and pull requests must pass the CI gate before merging.

### GitHub Repository Setup Guide
To configure this policy in GitHub:
1. Navigate to the repository on GitHub: `https://github.com/pritesh-4/Project_Zogan`.
2. Click **Settings** → **Branches** (or **Rules** → **Rulesets**).
3. Under **Branch protection rules**, click **Add branch ruleset** or **Add rule**.
4. Set **Branch name pattern** to `main`.
5. Enable the following protection policies:
   - **Require a pull request before merging**: Prevents direct pushes to `main`.
   - **Require status checks to pass before merging**:
     - Check **Require branches to be up to date before merging**.
     - In the search box, select the status check: **`Code Quality & Test Gate`** (Job name: `ci`).
   - **Do not allow force pushes**: Blocks force-pushes to `main`.
   - **Do not allow deletions**: Prevents accidental deletion of the `main` branch.
6. Click **Save changes** / **Create**.

With this policy active:
```text
feature/branch ──► Pull Request ──► GitHub Actions CI ──► Required Check (ci) PASS ──► Merge into main
```

---

## ⭐ Contributing & License

This project is built for **wildlife conservation, human-wildlife conflict mitigation, and community safety**.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch guidelines, setup instructions, and contribution best practices.

