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
# ⚖️ RULE-BASED RISK ENGINE 2.0 & THREAT ASSESSMENT CONFIGURATION (PHASE 7)
# =============================================================================
RISK_ENGINE_ENABLED: bool = True

# Score-to-level threshold mapping (0–100)
RISK_LEVEL_LOW_MAX: int = 24  # 0 - 24:   LOW (Monitoring)
RISK_LEVEL_MEDIUM_MAX: int = 49  # 25 - 49:  MEDIUM (Warning)
RISK_LEVEL_HIGH_MAX: int = 74  # 50 - 74:  HIGH (Local Alert)
# 75 - 100: CRITICAL (Emergency)

# Multi-Signal Factor Point Caps (Bounded contribution per signal)
MAX_ZONE_POINTS: int = 40
MAX_PROXIMITY_POINTS: int = 25
MAX_TREND_POINTS: int = 20
MAX_PERSISTENCE_POINTS: int = 20
MAX_GROUP_POINTS: int = 10
MAX_CONFIDENCE_POINTS: int = 10
MAX_DURATION_POINTS: int = 10
MAX_HISTORY_POINTS: int = 5

# Temporal Persistence Thresholds (Frames)
PERSISTENCE_CONFIRMATION_FRAMES: int = 5  # Frames required to confirm detection
PERSISTENCE_STRONG_FRAMES: int = 10  # Frames for elevated persistence evidence

# Evidence Requirements & Safety Safeguards
CRITICAL_REQUIRES_PERSISTENCE: bool = True  # Single-frame detection cannot produce CRITICAL
CRITICAL_MIN_CONFIDENCE: float = 0.70  # Minimum confidence required for CRITICAL
CRITICAL_MAX_DISTANCE_METERS: float = 300.0  # Max distance to protected zone for CRITICAL
SINGLE_FRAME_RISK_CAP: str = "MEDIUM"  # Risk level ceiling for unconfirmed single-frame detections

# Duration Thresholds in Risk Area (Seconds)
DURATION_ELEVATED_SECONDS: float = 10.0  # Dwell time in risk area before duration points apply
DURATION_CRITICAL_SECONDS: float = 30.0  # Sustained dwell time threshold

# Herd Configuration
HERD_ENABLED: bool = True
HERD_THRESHOLD: int = 3  # Count of elephants considered a herd elevation

# Movement Trend Configuration
MOVEMENT_RISK_ENABLED: bool = True
TREND_STABILITY_THRESHOLD_METERS: float = 15.0
TREND_WINDOW_SIZE: int = 4

# Alert Policy Configuration (Decoupled from Risk Engine)
ALERT_TRIGGER_RISK_LEVEL: str = "HIGH"
ALERT_ON_LEVELS: list = ["HIGH", "CRITICAL"]
ALERT_ESCALATION_ENABLED: bool = True  # Escalate alert when risk level increases
ESCALATION_COOLDOWN_BYPASS: bool = True  # Allow immediate alert if risk level escalates

# Stateful Event Lifecycle Configuration
EVENT_RESOLUTION_FRAMES: int = 30  # Consecutive missed frames before marking event RESOLVED (~1s at 30fps)
EVENT_RESOLUTION_SECONDS: float = 2.0  # Time without detection before closing event

# =============================================================================
# 🩺 SYSTEM HEALTH & OBSERVABILITY CONFIGURATION (PHASE 8)
# =============================================================================
HEALTH_MONITOR_ENABLED: bool = True

# Frame Freshness Thresholds (seconds since last successful frame)
HEALTH_FRESHNESS_DEGRADED_SECONDS: float = 1.5  # Freshness latency above this marks pipeline DEGRADED
HEALTH_FRESHNESS_OFFLINE_SECONDS: float = 5.0  # Freshness latency above this marks camera OFFLINE

# FPS Thresholds & Smoothing
HEALTH_FPS_MINIMUM_HEALTHY: float = 12.0  # FPS below this threshold marks system DEGRADED
HEALTH_FPS_EXPECTED: float = 30.0  # Expected baseline capture/processing frame rate
HEALTH_FPS_WINDOW_SIZE: int = 30  # Rolling frame window for stable FPS measurement

# Failure & Error Thresholds
HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES: int = 5  # Consecutive frame read drops before DEGRADED
HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES_OFFLINE: int = 20  # Read drops before marking camera OFFLINE
HEALTH_MAX_CONSECUTIVE_ERRORS_DEGRADED: int = 3  # Consecutive subsystem errors before DEGRADED
HEALTH_MAX_CONSECUTIVE_ERRORS_OFFLINE: int = 10  # Consecutive subsystem errors before OFFLINE
HEALTH_MAX_ERROR_HISTORY: int = 20  # Maximum bounded error records kept in memory

# Heartbeat & Logging
HEALTH_HEARTBEAT_TIMEOUT_SECONDS: float = 5.0  # Time without pipeline pulse before watchdog timeout
HEALTH_LOG_TRANSITIONS_ONLY: bool = True  # Log only when system health state changes

# =============================================================================
# 📊 LOGGING & OFFLINE QUEUE CONFIGURATION (PHASE 9)
# =============================================================================
LOG_DIR: str = "logs"
ALERT_LOG_FILE: str = os.path.join(LOG_DIR, "alerts.jsonl")
ALERT_QUEUE_FILE: str = os.path.join(LOG_DIR, "delivery_queue.jsonl")
OFFLINE_QUEUE_ENABLED: bool = True
ALERT_QUEUE_MAX_SIZE: int = 100  # Maximum bounded entries in the persistent queue
ALERT_RETRY_LIMIT: int = 3  # Maximum delivery retry attempts before marking FAILED
ALERT_RETRY_BASE_DELAY_SECONDS: float = 2.0  # Initial exponential backoff delay
ALERT_RETRY_BACKOFF_FACTOR: float = 2.0  # Exponential multiplier (2.0s, 4.0s, 8.0s)
ALERT_DELIVERY_TIMEOUT_SECONDS: float = 3.0  # Fast timeout so video processing is never blocked

# Camera Recovery Configuration
CAMERA_RECONNECT_ENABLED: bool = True
CAMERA_MAX_RECONNECT_ATTEMPTS: int = 3
CAMERA_RECONNECT_DELAY_SECONDS: float = 1.0

# Detector Recovery Configuration
DETECTOR_MAX_CONSECUTIVE_FAILURES: int = 3  # Consecutive exceptions before marking FAILED

# =============================================================================
# 🎨 DISPLAY COLORS (BGR format for OpenCV)
# =============================================================================
COLOR_ALERT_RED = (0, 0, 255)
COLOR_WARN_YELLOW = (0, 215, 255)
COLOR_SAFE_GREEN = (0, 255, 0)
COLOR_OTHER_OBJ = (255, 180, 0)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_OVERLAY_BG = (25, 25, 25)
