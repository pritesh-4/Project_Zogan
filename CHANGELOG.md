# 🐘 Project Zogan — Engineering Audit & Refactoring Changelog

All notable changes made during the deep engineering audit, refactoring, and AI capability upgrade are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased / Audit Refactor] - 2026-10-03

### 🧹 Repository Cleanup & Hygiene
- **Removed Tracked Build Artifacts**: Staged removal of 98+ experiment evaluation artifacts, plots, weight files, and confusion matrices tracked under `runs/` in Git despite `.gitignore`.
- **Removed Generated Test Images**: Removed committed preview images (`predicted_elephant.jpg`, `comparison_elephant.jpg`, `test_output_annotated.jpg`, `test_phase3_4_hud_preview.jpg`, `test_phase4_tracking_preview.jpg`) from repository root.
- **Removed Dead Scripts**:
  - Deleted `detect.py`: Obsolete single-image script with hardcoded paths, superseded by `ai/predict_custom.py`.
  - Deleted `camera.py`: Bare OpenCV test script without detection, superseded by `elephant_camera.py`.
- **Updated `.gitignore`**:
  - Explicitly ignored `predicted_*.jpg`, `comparison_*.jpg`, `test_output_*.jpg`, `*_hud_preview.jpg`, `*_tracking_preview.jpg`.
  - Explicitly ignored runtime JSONL event logs `logs/*.jsonl` while preserving `logs/.gitkeep`.
  - Ignored `runs/` and YOLO `.cache` files.

### 🏗️ Architecture & Modular Decomposition
- **Modularized God File (`elephant_camera.py`)**:
  - Decomposed 987-line monolith into dedicated, single-responsibility modules under `ai/`.
  - Maintained complete backward compatibility by re-exporting necessary configuration constants and function aliases (`resolve_active_model`, `draw_hud`, `draw_bounding_box`, `trigger_alert`).
- **Created `ai/detector.py`**:
  - Implemented `Detector` abstraction with dataclasses `Detection` and `DetectorMetadata`.
  - Encapsulates YOLO model loading, class verification, and inference, insulating the main orchestrator from framework APIs.
- **Created `ai/model_manager.py`**:
  - Centralized model resolution hierarchy (explicit override $\to$ fine-tuned model $\to$ fallback custom path $\to$ pretrained baseline).
  - Added model file validation (`verify_model`) ensuring non-empty file existence.
- **Created `ai/renderer.py`**:
  - Decoupled all visual rendering logic: bounding boxes, tracking trajectories, movement labels, responsive HUD, and alert banners.
- **Created `ai/__init__.py`**:
  - Established `ai` as a formal Python package with comprehensive `__all__` exports for clean external imports.
- **Centralized Configuration (`config.py`)**:
  - Consolidated scattered parameters across `elephant_camera.py` and `ai/` into a single documented configuration module.
  - Defined explicit project root, camera settings, model paths, detection thresholds, cooldowns, display colors, and risk parameters.

### 🐛 Bug Fixes & Logic Corrections
- **Fixed Static Simulated Coordinates Bug**:
  - In `elephant_camera.py`, simulated elephant coordinates previously remained identical every frame (`sim_elephant_lat = base_sim_lat`), causing geofence zone, distance, and approach trends to be permanently static during video feeds.
  - Implemented `SimulatedElephantCoordinates` with realistic spatial drift and course-change mechanics, enabling genuine dynamic geofence transitions and risk engine evaluation during simulation mode.
- **Dynamic Dataset Metrics in `ai/evaluate.py`**:
  - Replaced hardcoded string templates (`"68 images"`, `"73 unseen images"`, `"Parameters: 2,375,031"`) with dynamic values derived from actual evaluation metrics and dataset counts.
- **Dynamic Zone Radii in `ai/simulate_risk.py`**:
  - Replaced magic numbers (`500m`, `1200m`) with dynamic computations based on `config.BUFFER_RADIUS_METERS` and `config.VILLAGE_RADIUS_METERS`.
- **Windows Terminal UTF-8 Reconfiguration**:
  - Added `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` across `scripts/health_check.py` to eliminate `UnicodeEncodeError` when rendering checkmarks (`✓`) on Windows consoles.

### 🧪 Testing & CI Improvements
- **Created Diagnostic Health Check (`scripts/health_check.py`)**:
  - Fast, honest startup diagnostic that verifies Python runtime, OpenCV build, Ultralytics version, PyTorch execution provider, model weights, dataset splits, and Telegram status without fake "OK" output.
- **Configured Pytest in `pyproject.toml`**:
  - Added `[tool.pytest.ini_options]` with test discovery and warning filters for `pytest_asyncio` deprecation notices, achieving 100% clean test passes with 0 warnings.
- **Preserved 100% Test Pass Rate**:
  - All 81 tests across all test suites pass without regression:
    - Phase 2 (Persistence & Alerting): 100% pass
    - Phase 3.1 (Dataset Structure & Validator): 100% pass
    - Phase 3.2 (Data Integrity & No Leakage): 100% pass
    - Phase 3.4 (Custom Model Detection): 100% pass
    - Phase 4 (ByteTrack Tracking & Motion): 100% pass
    - Phase 5 (Geofencing & Risk Engine): 100% pass
    - Phase 6 (Alert Event Logging & Dispatcher): 100% pass
    - Alerts Pytest Suite: 100% pass

### 📝 Documentation & Truthfulness Corrections
- **Created `ARCHITECTURE.md`**:
  - Exhaustive system architecture document with Mermaid diagrams, modular component breakdown, and real vs. simulated boundary map.
- **Corrected `models/README.md`**:
  - Removed misleading "Active Production Model" claim, reframing `elephant_v1` accurately as the primary fine-tuned prototype for research and evaluation.
- **Deduplicated `README.md` Roadmap**:
  - Removed premature duplicate roadmap block (lines 450–492) that interrupted the flow between Phase 4 and Phase 5.
  - Linked Phase 4/5/6 roadmap items accurately to reflect implemented functionality.
  - Updated repository tree diagram in `README.md` to reflect new modules and removed dead scripts.
  - Updated Quick Start command to recommend `ai/predict_custom.py` instead of deleted `detect.py`.
  - Documented root cause of file count discrepancy (456 `.jpg` images + 3 `.gitkeep` files = 459 filesystem entries).
