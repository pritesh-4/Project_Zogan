"""
=============================================================================
🐘 PROJECT ZOGAN — RULE-BASED RISK ENGINE (PHASE 5)
=============================================================================

This module provides a transparent, rule-based risk evaluation engine that
computes an early-warning risk score (0–100) and maps it to actionable levels
(LOW, MEDIUM, HIGH, CRITICAL).

Pipeline Integration:
  Detections + Tracks + Geofencing -> ai/risk_engine.py -> HUD Overlay & Alerts

Inputs Considered:
  1. Current Zone (VILLAGE / BUFFER / FOREST / OUTSIDE)
  2. Distance to Protected / Sensitive Zone (meters)
  3. Geographic Approach Trend (APPROACHING / RECEDING / STABLE / UNKNOWN)
  4. Elephant Group Size (number of simultaneously tracked elephants)
  5. Detection Confidence (0.0 – 1.0)

⚠️ IMPORTANT SCIENTIFIC & ETHICAL DISCLAIMER:
  - This is an experimental prototype decision-support layer.
  - The risk scoring is NOT a scientifically validated behavioral model.
  - An elephant is NOT inherently dangerous simply because it exists.
  - Risk is defined strictly by configured spatial proximity to human habitations.
  - Real-world deployment requires local conservation authority calibration.
=============================================================================
"""

from collections import deque
from typing import Dict, List, Optional

# Import Zone Types from Geofencing
from ai.geofence import (
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
    ZONE_TYPE_MONITORING,
)

# Risk Level Designations
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"

VALID_RISK_LEVELS = [RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL]

# Movement Trend Designations
TREND_APPROACHING = "APPROACHING"
TREND_RECEDING = "RECEDING"
TREND_STABLE = "STABLE"
TREND_UNKNOWN = "UNKNOWN"

VALID_TRENDS = [TREND_APPROACHING, TREND_RECEDING, TREND_STABLE, TREND_UNKNOWN]


# =============================================================================
# 📈 MOVEMENT TREND TRACKER
# =============================================================================


class MovementTrendTracker:
    """
    Maintains a temporal history of distance measurements from a tracked elephant
    to a sensitive protected zone, and computes whether the elephant is
    APPROACHING, RECEDING, or STABLE relative to the protected area.
    """

    def __init__(
        self,
        window_size: int = 4,
        stability_threshold_meters: float = 15.0,
    ):
        self.window_size = max(2, window_size)
        self.stability_threshold = float(stability_threshold_meters)
        self.history: deque = deque(maxlen=self.window_size)

    def add_observation(self, distance_meters: float) -> str:
        """
        Records a new distance observation and returns the updated trend.
        """
        self.history.append(float(distance_meters))
        return self.get_trend()

    def get_trend(self) -> str:
        """
        Evaluates the approach trend across the observation history.
        Requires at least 2 observations; otherwise returns UNKNOWN.
        """
        if len(self.history) < 2:
            return TREND_UNKNOWN

        # Compare current distance with earliest in window
        d_initial = self.history[0]
        d_current = self.history[-1]
        delta = d_current - d_initial

        # If net displacement is within noise tolerance: STABLE
        if abs(delta) < self.stability_threshold:
            return TREND_STABLE

        # If distance is decreasing: APPROACHING
        if delta <= -self.stability_threshold:
            return TREND_APPROACHING

        # If distance is increasing: RECEDING
        return TREND_RECEDING

    def clear(self) -> None:
        """Resets distance history."""
        self.history.clear()


# =============================================================================
# ⚖️ RISK ASSESSMENT RESULT
# =============================================================================


class RiskAssessment:
    """
    Holds the complete evaluation output produced by the Risk Engine.
    """

    def __init__(
        self,
        score: int,
        level: str,
        zone: str,
        distance_to_protected: float,
        trend: str,
        group_size: int,
        confidence: float,
        breakdown: Dict[str, int],
        alert_recommended: bool,
    ):
        self.score = int(score)
        self.level = level
        self.zone = zone
        self.distance_to_protected = float(distance_to_protected)
        self.trend = trend
        self.group_size = int(group_size)
        self.confidence = float(confidence)
        self.breakdown = breakdown
        self.alert_recommended = alert_recommended

    def to_dict(self) -> dict:
        return {
            "risk_score": self.score,
            "risk_level": self.level,
            "zone": self.zone,
            "distance_to_protected_m": round(self.distance_to_protected, 1),
            "trend": self.trend,
            "group_size": self.group_size,
            "confidence": round(self.confidence, 3),
            "alert_recommended": self.alert_recommended,
            "breakdown": self.breakdown,
        }

    def __repr__(self) -> str:
        return (
            f"RiskAssessment(level='{self.level}', score={self.score}/100, "
            f"zone='{self.zone}', dist={self.distance_to_protected:.0f}m, "
            f"trend='{self.trend}', group={self.group_size})"
        )


# =============================================================================
# 🧠 RULE-BASED RISK ENGINE
# =============================================================================


class RiskEngine:
    """
    Transparent, configurable rule-based risk evaluation engine.
    """

    def __init__(
        self,
        low_max: int = 24,
        medium_max: int = 49,
        high_max: int = 74,
        alert_on_levels: Optional[List[str]] = None,
    ):
        self.low_max = low_max
        self.medium_max = medium_max
        self.high_max = high_max
        self.alert_on_levels = set(alert_on_levels or [RISK_HIGH, RISK_CRITICAL])

    def evaluate(
        self,
        zone: str,
        distance_to_protected: float,
        trend: str = TREND_UNKNOWN,
        group_size: int = 1,
        confidence: float = 1.0,
    ) -> RiskAssessment:
        """
        Computes a numerical risk score (0–100) and maps it to a risk level.
        """
        zone_clean = zone.upper()
        trend_clean = trend.upper()
        dist = max(0.0, float(distance_to_protected))
        group = max(1, int(group_size))
        conf = min(1.0, max(0.0, float(confidence)))

        breakdown: Dict[str, int] = {}

        # 1. Zone Severity Score (Max: 40 points)
        if zone_clean in (ZONE_TYPE_PROTECTED, "PROTECTED", "HUMAN_ZONE"):
            zone_pts = 40
        elif zone_clean in (ZONE_TYPE_WARNING, "WARNING", "BUFFER"):
            zone_pts = 20
        elif zone_clean in (ZONE_TYPE_MONITORING, "MONITORING", "FOREST"):
            zone_pts = 5
        else:
            zone_pts = 0
        breakdown["zone_points"] = zone_pts

        # 2. Proximity to Protected Area (Max: 25 points)
        if dist <= 0.0 or zone_clean in (ZONE_TYPE_PROTECTED, "PROTECTED"):
            dist_pts = 25
        elif dist <= 200.0:
            dist_pts = 20
        elif dist <= 500.0:
            dist_pts = 15
        elif dist <= 1000.0:
            dist_pts = 5
        else:
            dist_pts = 0
        breakdown["proximity_points"] = dist_pts

        # 3. Geographic Approach Trend (Max: 20 points)
        if trend_clean == TREND_APPROACHING:
            trend_pts = 20
        elif trend_clean in (TREND_STABLE, TREND_UNKNOWN):
            trend_pts = 5
        elif trend_clean == TREND_RECEDING:
            trend_pts = 0
        else:
            trend_pts = 5
        breakdown["trend_points"] = trend_pts

        # 4. Group Size / Herd Count (Max: 10 points)
        if group >= 4:
            group_pts = 10
        elif group >= 2:
            group_pts = 5
        else:
            group_pts = 2
        breakdown["group_points"] = group_pts

        # 5. Detection Confidence (Max: 10 points)
        if conf >= 0.90:
            conf_pts = 10
        elif conf >= 0.80:
            conf_pts = 7
        elif conf >= 0.70:
            conf_pts = 5
        else:
            conf_pts = 0
        breakdown["confidence_points"] = conf_pts

        # Aggregate total score clamped to [0, 100]
        total_score = sum(breakdown.values())
        total_score = min(100, max(0, total_score))

        # Map score to risk level
        if total_score <= self.low_max:
            level = RISK_LOW
        elif total_score <= self.medium_max:
            level = RISK_MEDIUM
        elif total_score <= self.high_max:
            level = RISK_HIGH
        else:
            level = RISK_CRITICAL

        # Determine if alert threshold is satisfied
        alert_recommended = level in self.alert_on_levels

        return RiskAssessment(
            score=total_score,
            level=level,
            zone=zone_clean,
            distance_to_protected=dist,
            trend=trend_clean,
            group_size=group,
            confidence=conf,
            breakdown=breakdown,
            alert_recommended=alert_recommended,
        )
