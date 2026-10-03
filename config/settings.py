"""
=============================================================================
🐘 PROJECT ZOGAN — CENTRAL CONFIGURATION
=============================================================================

All configurable parameters for detection, tracking, geofencing, risk
assessment, and alerting.

⚠️ IMPORTANT NOTICES:
  - Camera coordinates below are SIMULATED / DEMO values.
  - Risk scores are prototype decision-support rules, NOT a validated
    wildlife behavioral model.
  - Camera location does NOT equal elephant location.
=============================================================================
"""

import os
from pathlib import Path

# =============================================================================
# 📁 PROJECT PATHS
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# =============================================================================
# 🤖 MODEL CONFIGURATION
# =============================================================================
CUSTOM_MODEL_PATH: str = "models/elephant_v1/best.pt"
FALLBACK_CUSTOM_PATH: str = "runs/detect/elephant_v1/weights/best.pt"
PRETRAINED_MODEL_PATH: str = "models/yolo26n.pt"
USE_CUSTOM_MODEL: bool = True

# Target class name to monitor
TARGET_CLASS: str = "elephant"

# =============================================================================
# 🎯 DETECTION CONFIGURATION
# =============================================================================
# Minimum confidence required to accept a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD: float = 0.70

# Number of consecutive frames required before confirming a detection
REQUIRED_DETECTIONS: int = 5

# Minimum bounding box area in pixels (filters tiny spurious detections)
MIN_BOX_AREA_PIXELS: int = 400

# =============================================================================
# 🚨 ALERT CONFIGURATION
# =============================================================================
# Seconds to wait before allowing another alert (prevents spam)
ALERT_COOLDOWN_SECONDS: int = 30

# Duration in seconds to display visual alert banner on screen
ALERT_BANNER_DURATION_SECONDS: float = 4.0

# =============================================================================
# 📹 CAMERA CONFIGURATION
# =============================================================================
# Default webcam index (0 is usually the built-in or primary USB camera)
CAMERA_INDEX: int = 0

# =============================================================================
# 🎯 TRACKING CONFIGURATION
# =============================================================================
TRACKER_CONFIG: str = "bytetrack.yaml"
MOVEMENT_THRESHOLD_PIXELS: float = 10.0  # Minimum pixels to count as movement
MAX_POSITION_HISTORY: int = 20  # Bounded history per track
MAX_LOST_FRAMES: int = 30  # Frames before evicting lost track (~1s at 30fps)
SMOOTHING_WINDOW_FRAMES: int = 5  # Multi-frame smoothing window

# =============================================================================
# 📍 SIMULATED CAMERA LOCATION
# =============================================================================
# Demo coordinates — NOT a real user's private location
CAMERA_LATITUDE: float = 20.123456
CAMERA_LONGITUDE: float = 85.123456
CAMERA_HEADING_DEGREES: float = 0.0  # North (for future orientation modeling)

# System Geolocation Mode: True = software simulation of elephant coordinates
DEFAULT_SIMULATION_MODE: bool = True

# =============================================================================
# 🗺️ GEOFENCING ZONE DEFINITIONS
# =============================================================================
# Protected Village Center (approx. 600m South-West of camera)
VILLAGE_CENTER_LAT: float = 20.119000
VILLAGE_CENTER_LON: float = 85.119000
VILLAGE_RADIUS_METERS: float = 300.0

# Buffer Zone (concentric warning perimeter from village center)
BUFFER_RADIUS_METERS: float = 800.0

# Forest Monitoring Zone (concentric monitoring perimeter)
FOREST_RADIUS_METERS: float = 2000.0

# =============================================================================
# ⚖️ RULE-BASED RISK ENGINE CONFIGURATION
# =============================================================================
# Score-to-level threshold mapping (0–100)
RISK_LEVEL_LOW_MAX: int = 24  # 0 - 24:   LOW (Monitoring)
RISK_LEVEL_MEDIUM_MAX: int = 49  # 25 - 49:  MEDIUM (Warning)
RISK_LEVEL_HIGH_MAX: int = 74  # 50 - 74:  HIGH (Local Alert)
# 75 - 100: CRITICAL (Emergency)

# Movement trend stability threshold in meters
TREND_STABILITY_THRESHOLD_METERS: float = 15.0

# Number of distance observations for trend evaluation
TREND_WINDOW_SIZE: int = 4

# Minimum risk level to trigger alerts
ALERT_TRIGGER_RISK_LEVEL: str = "HIGH"

# =============================================================================
# 📊 LOGGING CONFIGURATION
# =============================================================================
LOG_DIR: str = "logs"
ALERT_LOG_FILE: str = os.path.join(LOG_DIR, "alerts.jsonl")

# =============================================================================
# 🎨 DISPLAY COLORS (BGR format for OpenCV)
# =============================================================================
COLOR_ALERT_RED = (0, 0, 255)
COLOR_WARN_YELLOW = (0, 215, 255)
COLOR_SAFE_GREEN = (0, 255, 0)
COLOR_OTHER_OBJ = (255, 180, 0)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_OVERLAY_BG = (25, 25, 25)
