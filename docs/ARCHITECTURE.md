# 🐘 Project Zogan — System Architecture & Engineering Design

**Project**: Real-Time Human-Elephant Conflict (HEC) Early Warning System  
**Version**: 1.0.0-audit-refactor  
**Status**: Research & Field Evaluation Prototype (Non-Production)  

---

## 1. System Overview & Core Philosophy

Project Zogan is designed to mitigate human-elephant conflict along forest-village fringes by combining edge computer vision (YOLO), multi-target tracking (ByteTrack), spatial intelligence (geofencing and geodesic distance analysis), and multi-channel alerting (JSON Lines event logging and Telegram messaging).

### Engineering Truthfulness Protocol
* **Computer Vision**: Real YOLO models (Ultralytics YOLO26n base + custom fine-tuned `elephant_v1` on 456 images).
* **Multi-Object Tracking**: Real ByteTrack tracking assigning frame-to-frame session IDs.
* **Spatial Intelligence**: **Simulated** in software. An RGB monocular camera cannot compute real-world GPS coordinates without calibrated depth/ranging sensors. Camera GPS $\neq$ Elephant GPS.
* **Risk Engine**: Deterministic, rule-based decision support heuristics. **Not** a biologically validated animal behavior model.
* **Alert System**: Real JSON Lines event logging and real asynchronous Telegram Bot API delivery (fail-safe, non-crashing).

---

## 2. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph Input ["Visual Input Layer"]
        CAM["Live Camera (Webcam / RTSP / USB)"] --> ORCH
        VID["Recorded Video File (.mp4 / .avi)"] --> ORCH
    end

    subgraph Core ["Orchestration & Detection (scripts/run_camera.py)"]
        ORCH["Main Loop Orchestrator"]
        MM["ai.model_manager<br/>(resolve_model_path)"] --> ORCH
        CFG["config/<br/>(Central Settings)"] --> ORCH
        DET["ai.detector<br/>(Detector Abstraction)"] --> ORCH
        YOLO["Ultralytics YOLO<br/>(elephant_v1 / models/yolo26n.pt)"] --- DET
    end

    subgraph TrackingLayer ["Tracking & Motion Analysis"]
        ORCH --> TRK["ai.tracking<br/>(ElephantTracker)"]
        TRK --> BT["ByteTrack Algorithm"]
        TRK --> MOT["Trajectory History & Direction Estimation<br/>(Image Space: RIGHT/LEFT/UP/DOWN)"]
    end

    subgraph SpatialLayer ["Spatial & Risk Evaluation Layer"]
        ORCH --> SIM["SimulatedElephantCoordinates<br/>(Dynamic Spatial Drift)"]
        SIM --> GEO["ai.geofence<br/>(GeoZone & Haversine Distance)"]
        GEO --> ZON["Zone Classification<br/>(Village / Buffer / Forest / Outside)"]
        ZON --> RISK["ai.risk_engine<br/>(RiskEngine & MovementTrendTracker)"]
        RISK --> EVAL["Risk Score (0-100) & Level<br/>(LOW / MEDIUM / HIGH / CRITICAL)"]
    end

    subgraph VisualLayer ["Visualization & HUD"]
        ORCH --> REN["ai.renderer"]
        REN --> HUD["Responsive HUD Bar<br/>(FPS, Status, Risk, Geo Mode)"]
        REN --> BOX["Bounding Boxes & Track Trajectories"]
        REN --> BAN["Emergency Alert Banner"]
    end

    subgraph AlertLayer ["Alerting & Notification Subsystem"]
        ORCH --> GATE{"Persistence & Cooldown Gate<br/>(>= 5 frames, 30s cooldown)"}
        GATE -- Confirmed --\> DISP["alerts.dispatcher<br/>(dispatch_alert)"]
        DISP --> LOG["alerts.event_logger<br/>(logs/alerts.jsonl)"]
        DISP --> TG["alerts.telegram<br/>(Telegram Bot API)"]
    end
```

---

## 3. Modular Architecture Breakdown

Prior to this engineering refactor, the orchestrator was a monolithic 987-line God Script mixing model loading, inference, tracking, geofencing, rendering, alerting, and configuration. The system is now structured into modular, decoupled packages:

### A. Central Configuration (`config/settings.py`)
Single source of truth for all operating parameters:
* **Model settings**: Model paths (`models/elephant_v1/best.pt`, `models/yolo26n.pt`), fallback paths, target class.
* **Detection parameters**: Confidence threshold (`0.70`), required detection persistence (`5` frames), minimum pixel area (`400` px).
* **Alert parameters**: Cooldown duration (`30` seconds), banner display duration (`4.0` seconds).
* **Geolocation defaults**: Simulated camera coordinates (`20.123456, 85.123456`), village center, zone radii (Village: 300m, Buffer: 800m, Forest: 2000m).
* **Risk parameters**: Threshold mappings (LOW $\le 24$, MEDIUM $\le 49$, HIGH $\le 74$, CRITICAL $\ge 75$), approach trend stability margin (`15.0` m).
* **UI Palette**: Standardized BGR color tuples for HUD and alerts.

### B. AI Core Package (`ai/`)
1. **`ai.detector.Detector`**: Pure abstraction for model loading, validation of class mappings, and bounding-box inference. Completely decouples the camera orchestrator from YOLO-specific APIs.
2. **`ai.model_manager`**: Handles model resolution hierarchy (explicit override $\to$ fine-tuned model $\to$ fallback custom path $\to$ pretrained COCO baseline) and file integrity verification.
3. **`ai.renderer`**: Clean OpenCV drawing functions for bounding boxes, movement sub-labels, multi-point trajectories, responsive top status HUD, and prominent alert banners.
4. **`ai.tracking.ElephantTracker`**: Wraps ByteTrack multi-object tracking, maintains bounded centroid position history (max 20 points), estimates 2D image-space movement with anti-jitter deadbands, and tracks active targets.
5. **`ai.geofence.GeoZone`**: Implements Great-Circle Haversine geodesic distance calculations and point-in-zone classification for concentric circular zones and arbitrary polygonal perimeters.
6. **`ai.risk_engine.RiskEngine`**: Multi-factor scoring engine evaluating zone proximity, approach trend (APPROACHING / RECEDING / STABLE), herd group size, and detection confidence into a bounded 0–100 integer score.
7. **`ai.simulate_risk`**: Standalone simulation test harness verifying risk engine transitions across approaching, receding, and stable animal trajectories.
8. **`ai.evaluate`**: Generates precision, recall, and mAP metrics against held-out validation and test sets with dynamically calculated dataset statistics.

### C. Alert Subsystem (`alerts/`)
1. **`alerts.models.AlertEvent`**: Strict dataclass representing an incident. Emits clean JSON dictionaries without fabricated or placeholder values for missing optional fields.
2. **`alerts.event_logger.AlertEventLogger`**: Thread-safe atomic JSON Lines append logger (`logs/alerts.jsonl`) with resilient directory creation and corrupted-line tolerance.
3. **`alerts.telegram`**: Asynchronous/fail-safe Telegram Bot API client. Sanitizes environment credentials, formats clean markdown alert cards, and guarantees zero crashes upon network dropouts.
4. **`alerts.dispatcher.dispatch_alert`**: Single unified entry point coordinating local logging and remote dispatch based on priority thresholds.
5. **`alerts.history` & `alerts.history_cli`**: Query utilities for filtering, counting, and viewing historical alert records.

### D. System Diagnostics (`scripts/health_check.py`)
A fast pre-flight diagnostic tool checking Python runtime, OpenCV build, Ultralytics version, PyTorch execution provider (CPU/CUDA), model file presence, dataset split availability, and Telegram configuration.

---

## 4. Real vs. Simulated Boundary

| Component | Status | Reality vs. Simulation Boundary |
|:---|:---:|:---|
| **YOLO Elephant Detection** | **REAL** | Runs real neural network inference on RGB frames using fine-tuned weights (`elephant_v1`). |
| **ByteTrack Tracking** | **REAL** | Bipartite matching associating bounding boxes frame-to-frame. |
| **Camera GPS Coordinates** | **SIMULATED** | Pre-configured demo coordinates (`20.123456, 85.123456`). In production, this requires an active NMEA GPS receiver. |
| **Elephant GPS Coordinates** | **SIMULATED** | Monocular RGB cameras cannot measure absolute latitude/longitude. Software uses `SimulatedElephantCoordinates` with spatial drift to demonstrate risk engine behavior. |
| **Geofencing Engine** | **REAL CODE** | Geodesic math (Haversine) is genuine; coordinates evaluated are simulated. |
| **Risk Scoring Engine** | **PROTOTYPE** | Deterministic heuristics (distance + trend + herd size + confidence). Decision support only. |
| **Telegram Delivery** | **REAL** | Real HTTPS POST requests to Telegram Bot API when configured via environment variables. |
| **Event Logging** | **REAL** | Appends valid structured JSONL records to local disk. |

---

## 5. Security & Credentials Architecture

* **Zero Hardcoded Secrets**: No bot tokens or chat IDs exist in source code.
* **Environment-Based Config**: Managed via `.env` (gitignored) and `.env.example` template.
* **Template Token Sanitization**: Placeholder strings such as `YOUR_BOT_TOKEN_HERE` or `123456789:ABC...` are automatically detected and rejected to prevent spurious network calls.
* **Automated CI Secret Detection**: GitHub Actions CI validates with regex scans that no credentials or private keys are ever committed to the repository.

---

## 6. Target Hardware Roadmap

* **Phase 1–6 (Current)**: Local workstation / Laptop / USB Webcam / Video Replay.
* **Phase 7 (Edge AI)**:
  * Embedded SBC: NVIDIA Jetson Orin Nano (8GB) or Raspberry Pi 5 with AI Accelerator (Hailo-8 / Coral TPU).
  * Power Subsystem: 50W Solar Panel + 12V LiFePO4 battery pack with MPPT solar charge controller.
  * Connectivity: 4G LTE Cat-M1 / NB-IoT modem or Long-Range LoRaWAN node for off-grid forest perimeters.
  * Sensor Fusion: Dual RGB + Long-Wave Infrared (LWIR) thermal camera for nocturnal elephant movement.
