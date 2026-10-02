"""
=============================================================================
🐘 PROJECT ZOGAN — AI MODULE PACKAGE
=============================================================================

Computer vision, tracking, geofencing, and risk assessment components.
=============================================================================
"""

from ai.detector import Detection, Detector, DetectorMetadata
from ai.geofence import (
    GeoZone,
    calculate_distance,
    classify_zone,
    create_default_zones,
    get_distance_to_protected_zone,
)
from ai.model_manager import resolve_model_path, verify_model
from ai.renderer import draw_alert_banner, draw_bounding_box, draw_hud
from ai.risk_engine import (
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    MovementTrendTracker,
    RiskAssessment,
    RiskEngine,
)
from ai.tracking import (
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_STATIONARY,
    DIRECTION_UNKNOWN,
    DIRECTION_UP,
    ElephantTracker,
    TrackedElephant,
    calculate_center,
    estimate_direction,
)

__all__ = [
    # Detector
    "Detector",
    "Detection",
    "DetectorMetadata",
    # Model Manager
    "resolve_model_path",
    "verify_model",
    # Renderer
    "draw_bounding_box",
    "draw_hud",
    "draw_alert_banner",
    # Tracking
    "ElephantTracker",
    "TrackedElephant",
    "calculate_center",
    "estimate_direction",
    "DIRECTION_RIGHT",
    "DIRECTION_LEFT",
    "DIRECTION_UP",
    "DIRECTION_DOWN",
    "DIRECTION_STATIONARY",
    "DIRECTION_UNKNOWN",
    # Geofence
    "GeoZone",
    "calculate_distance",
    "classify_zone",
    "get_distance_to_protected_zone",
    "create_default_zones",
    # Risk Engine
    "RiskEngine",
    "MovementTrendTracker",
    "RiskAssessment",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
]
