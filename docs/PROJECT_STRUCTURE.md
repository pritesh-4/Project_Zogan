# 🐘 Project Zogan — Repository Structure & File Organization

This document details the standardized directory layout, package responsibilities, and file organization for **Project Zogan**.

---

## 📁 Repository Overview

```text
Elephant_detector/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions continuous integration pipeline
│
├── ai/                            # AI/ML core detection, tracking, spatial, and training modules
│   ├── __init__.py                # Package exports and module definitions
│   ├── compare_models.py          # Side-by-side inference comparison utility
│   ├── detector.py                # Detector abstraction and dataclass interfaces
│   ├── evaluate.py                # Model evaluation with dynamic dataset calculation
│   ├── geofence.py                # Haversine geodesic distance and zone classification
│   ├── model_manager.py           # Model path resolution, fallback, and validation
│   ├── predict_custom.py          # Single-image custom model prediction CLI
│   ├── renderer.py                # Visual rendering: bounding boxes, HUD, alerts
│   ├── risk_engine.py             # Rule-based early warning risk assessment heuristics
│   ├── simulate_risk.py           # Spatial trajectory and approach risk simulation demo
│   ├── tracker.py                 # Tracking alias module exposing tracker interface
│   ├── tracking.py                # ByteTrack multi-object tracking and movement estimation
│   ├── train.py                   # Transfer learning fine-tuning pipeline
│   ├── validate_dataset.py        # Dataset structure and annotation integrity validator
│   └── visualize_dataset.py       # Ground-truth annotation visualizer
│
├── alerts/                        # Alerting and incident notification subsystem
│   ├── __init__.py                # Package exports for alerts layer
│   ├── dispatcher.py              # Unified dispatcher (local JSON Lines + Telegram)
│   ├── event_logger.py            # Append-only JSON Lines event logger (logs/alerts.jsonl)
│   ├── history.py                 # Incident query, filtering, and latest-alert reader
│   ├── history_cli.py             # Command-line inspection tool for alert events
│   ├── models.py                  # Structured AlertEvent model and factory function
│   └── telegram.py                # Telegram Bot API notification client
│
├── config/                        # Application configuration and settings
│   ├── __init__.py                # Re-exports settings for convenient access
│   ├── settings.py                # Central system parameters, thresholds, and paths
│   └── zones.example.yaml         # Example YAML schema for spatial zone perimeters
│
├── datasets/                      # Training, validation, and evaluation datasets
│   └── elephant/
│       ├── images/
│       │   ├── train/             # 315 training images
│       │   ├── val/               # 68 validation images
│       │   └── test/              # 73 test images
│       ├── labels/
│       │   ├── train/             # 315 YOLO annotation text files
│       │   ├── val/               # 68 YOLO annotation text files
│       │   └── test/              # 73 YOLO annotation text files
│       ├── data.yaml              # Ultralytics dataset configuration file
│       ├── metadata.json          # Dataset manifest and split provenance
│       └── README.md              # Dataset documentation and YOLO annotation guide
│
├── docs/                          # Comprehensive project documentation
│   ├── ARCHITECTURE.md            # System architecture and engineering specifications
│   ├── AUDIT_REPORT.md            # Comprehensive engineering audit report
│   ├── CHANGELOG.md               # Version changelog and migration notes
│   ├── CONTRIBUTING.md            # Contributor guidelines and quality standards
│   └── PROJECT_STRUCTURE.md       # Repository layout and module responsibilities (this file)
│
├── logs/                          # Runtime log directories
│   ├── .gitkeep                   # Directory placeholder
│   └── alerts.jsonl               # Runtime alert event log (gitignored)
│
├── models/                        # Trained model checkpoints and baseline weights
│   ├── README.md                  # Model catalog, evaluation benchmarks, and deployment notes
│   ├── yolo26n.pt                 # Pretrained COCO baseline weights
│   └── elephant_v1/
│       ├── best.pt                # Custom fine-tuned elephant detector weights
│       └── README.md              # Version-specific training and performance notes
│
├── scripts/                       # Executable entrypoints and CLI utilities
│   ├── __init__.py                # Package initialization for scripts
│   ├── detect_image.py            # User-friendly single-image detection CLI
│   ├── health_check.py            # Comprehensive system diagnostic and pre-flight check
│   └── run_camera.py              # Main real-time detection, tracking & risk orchestrator
│
├── tests/                         # Automated test suite and test fixtures
│   ├── __init__.py                # Tests package initialization
│   ├── fixtures/
│   │   └── elephant.jpg           # Standard test fixture image
│   ├── test_alerts.py             # Alert subsystem pytest suite
│   ├── test_phase2.py             # Phase 2 test suite (Persistence & Cooldown)
│   ├── test_phase3_1.py           # Phase 3.1 test suite (Dataset Structure & Validator)
│   ├── test_phase3_2.py           # Phase 3.2 test suite (Data Integrity & Splits)
│   ├── test_phase3_4.py           # Phase 3.4 test suite (Custom Model Pipeline)
│   ├── test_phase4.py             # Phase 4 test suite (ByteTrack Tracking & Motion)
│   ├── test_phase5.py             # Phase 5 test suite (Geofencing & Risk Engine)
│   └── test_phase6.py             # Phase 6 test suite (Event Logging & Telegram Dispatcher)
│
├── .env.example                   # Environment variable template for credentials
├── .gitignore                     # Git exclusion rules
├── pyproject.toml                 # Tool configurations (Ruff, Pytest)
├── README.md                      # Primary project overview and quickstart guide
├── requirements.txt               # Runtime production dependencies
└── requirements-dev.txt           # Testing and development dependencies
```

---

## 🏛️ Directory Responsibilities

### `ai/` — AI/ML Core Modules
Contains all algorithmic components for computer vision inference, object tracking, spatial geofencing, and risk evaluation:
* **Detection (`detector.py`, `model_manager.py`)**: Encapsulates YOLO model loading, fallback resolution, and inference abstractions.
* **Tracking (`tracker.py`, `tracking.py`)**: Multi-object ByteTrack tracking, trajectory history, and image-space movement estimation.
* **Spatial & Risk (`geofence.py`, `risk_engine.py`, `simulate_risk.py`)**: Haversine distance calculations, multi-tier zone classification (Village, Buffer, Forest), and deterministic rule-based threat evaluation.
* **Dataset & Training Utilities (`train.py`, `evaluate.py`, `validate_dataset.py`, `visualize_dataset.py`)**: End-to-end model training, dynamic metric evaluation, dataset integrity verification, and bounding box visualization.

### `alerts/` — Alerting & Incident Notification
Provides structured incident event generation and multi-channel delivery:
* **Data Modeling (`models.py`)**: Dataclass-based `AlertEvent` schema with serialization and deserialization.
* **Logging (`event_logger.py`, `history.py`, `history_cli.py`)**: Append-only JSON Lines logging with crash resilience and query utilities.
* **Notification (`telegram.py`, `dispatcher.py`)**: Telegram Bot API integration with graceful unconfigured fallback and unified multi-channel dispatch.

### `config/` — Configuration & Settings
Houses central configuration:
* **Settings (`settings.py`, `__init__.py`)**: Parameterized thresholds, camera settings, zone radii, and file paths. Re-exported at package root (`import config`).
* **Zone Schema (`zones.example.yaml`)**: Example declarative YAML configuration for spatial boundaries.

### `datasets/` — Training & Validation Data
Maintains dataset files isolated by domain:
* **Images & Labels (`images/`, `labels/`)**: Strictly disjoint train, validation, and test splits.
* **Dataset Config (`data.yaml`, `metadata.json`, `README.md`)**: Ultralytics YOLO configuration, provenance tracking, and annotation guidelines.

### `docs/` — Project Documentation
Project-wide documentation migrated out of repository root:
* `ARCHITECTURE.md`: Technical architecture and dataflow diagrams.
* `AUDIT_REPORT.md`: Comprehensive engineering audit findings and remediation records.
* `CHANGELOG.md`: Chronological log of changes and refactoring phases.
* `CONTRIBUTING.md`: Development guidelines, pull request protocols, and code quality standards.
* `PROJECT_STRUCTURE.md`: Repository layout and package responsibilities.

### `models/` — Trained Model Checkpoints
Stores model weights and evaluation documentation:
* `yolo26n.pt`: Pretrained baseline YOLO model.
* `elephant_v1/best.pt`: Custom fine-tuned weights for elephant detection.
* `README.md`: Version specifications, benchmark numbers, and deployment guides.

### `scripts/` — Executable Utilities & Entrypoints
Command-line programs and application entrypoints:
* `run_camera.py`: Primary application entrypoint for webcam or video file surveillance.
* `detect_image.py`: Command-line utility to run inference on single images.
* `health_check.py`: System diagnostic script verifying environment and dependencies.

### `tests/` — Automated Test Suite
Automated quality gates and test fixtures:
* Phase test suites (`test_phase2.py` through `test_phase6.py`): Comprehensive behavioral verification.
* Subsystem tests (`test_alerts.py`): Pytest-based unit and integration test suite.
* `fixtures/`: Test media fixtures (`elephant.jpg`).
