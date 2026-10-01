"""
=============================================================================
🐘 PROJECT ZOGAN — CONFIGURATION SETTINGS (PHASE 5)
=============================================================================

Central configuration for camera geolocation, geofenced zones, and
rule-based risk assessment parameters.

⚠️ IMPORTANT SAFETY & PROTOCOL NOTICE:
  - The camera location coordinates below are SIMULATED / DEMO coordinates.
  - They do NOT represent a real user's private location.
  - The risk scores and levels are prototype decision-support rules,
    NOT a scientifically validated wildlife behavioral model.
  - Camera location does NOT equal elephant location. An RGB camera cannot
    derive real-world GPS coordinates without calibrated depth/ranging sensors.
=============================================================================
"""

# =============================================================================
# 📍 SIMULATED CAMERA LOCATION
# =============================================================================
# Demo coordinates situated near a wildlife monitoring perimeter
CAMERA_LATITUDE: float = 20.123456
CAMERA_LONGITUDE: float = 85.123456
CAMERA_HEADING_DEGREES: float = 0.0  # North (for future orientation modeling)

# System Geolocation Mode: True indicates software simulation of elephant coordinates
DEFAULT_SIMULATION_MODE: bool = True

# =============================================================================
# 🗺️ GEOFENCING ZONE DEFINITIONS
# =============================================================================
# Geographic zones configured relative to the monitoring post and protected village.
# Zone Types:
#   - PROTECTED  / VILLAGE: Human settlement zone (High priority protection)
#   - WARNING    / BUFFER:  Transition zone between forest and village
#   - MONITORING / FOREST:  Natural habitat / forest monitoring zone

# Protected Village Center (approx. 600m South-West of camera)
VILLAGE_CENTER_LAT: float = 20.119000
VILLAGE_CENTER_LON: float = 85.119000
VILLAGE_RADIUS_METERS: float = 300.0

# Buffer Zone (concentric warning perimeter extending 800m from village center)
BUFFER_RADIUS_METERS: float = 800.0

# Forest Monitoring Zone (concentric monitoring perimeter extending 2000m)
FOREST_RADIUS_METERS: float = 2000.0

# =============================================================================
# ⚖️ RULE-BASED RISK ENGINE CONFIGURATION
# =============================================================================
# Threshold definitions for mapping numerical risk scores (0–100) to levels
RISK_LEVEL_LOW_MAX: int = 24  # 0 - 24:   LOW (Monitoring)
RISK_LEVEL_MEDIUM_MAX: int = 49  # 25 - 49:  MEDIUM (Warning)
RISK_LEVEL_HIGH_MAX: int = 74  # 50 - 74:  HIGH (Local Alert)
# 75 - 100: CRITICAL (Emergency Alert)

# Movement trend stability threshold in meters
# Distance changes smaller than this are classified as STABLE (anti-noise)
TREND_STABILITY_THRESHOLD_METERS: float = 15.0

# Number of distance observations required to evaluate approach/recede trend
TREND_WINDOW_SIZE: int = 4

# Minimum risk level required to trigger local warning/alarm alerts
# Options: "LOW", "MEDIUM", "HIGH", "CRITICAL"
ALERT_TRIGGER_RISK_LEVEL: str = "HIGH"
