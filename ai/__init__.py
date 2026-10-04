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
    EVENT_STATE_CONFIRMED,
    EVENT_STATE_CRITICAL,
    EVENT_STATE_DETECTED,
    EVENT_STATE_HIGH_RISK,
    EVENT_STATE_IDLE,
    EVENT_STATE_RESOLVED,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    TREND_APPROACHING,
    TREND_RECEDING,
    TREND_STABLE,
    TREND_UNKNOWN,
    AlertDecision,
    AlertPolicy,
    EventLifecycleManager,
    EventTransition,
    MovementTrendTracker,
    RiskAssessment,
    RiskEngine,
    ThreatAssessment,
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
    # Risk Engine 2.0
    "ThreatAssessment",
    "RiskAssessment",
    "RiskEngine",
    "MovementTrendTracker",
    "AlertPolicy",
    "AlertDecision",
    "EventLifecycleManager",
    "EventTransition",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "TREND_APPROACHING",
    "TREND_RECEDING",
    "TREND_STABLE",
    "TREND_UNKNOWN",
    "EVENT_STATE_IDLE",
    "EVENT_STATE_DETECTED",
    "EVENT_STATE_CONFIRMED",
    "EVENT_STATE_HIGH_RISK",
    "EVENT_STATE_CRITICAL",
    "EVENT_STATE_RESOLVED",
]
