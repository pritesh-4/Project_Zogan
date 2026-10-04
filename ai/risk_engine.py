"""
=============================================================================
🐘 PROJECT ZOGAN — INTELLIGENT THREAT ASSESSMENT & RISK ENGINE 2.0 (PHASE 7)
=============================================================================

This module provides a robust, explainable, rule-based threat assessment layer
that combines multiple signals to evaluate human-elephant conflict (HEC) risk:

Pipeline Architecture:
  Detector (YOLO)
       ↓
  Tracker (ByteTrack)
       ↓
  Context Extraction (Geofence & Geodesic Distances)
       ↓
  Risk Engine 2.0 (ThreatAssessment & Multi-Signal Rules)
       ↓
  Event Lifecycle Manager (In-Memory State Machine)
       ↓
  Alert Policy (Separates Assessment from Dispatch)
       ↓
  Unified Alert Dispatcher (JSONL Logging & Telegram Delivery)

Signals Evaluated:
  1. Zone Membership (VILLAGE/PROTECTED > BUFFER/WARNING > FOREST/MONITORING > OUTSIDE)
  2. Proximity to Protected Boundary (meters)
  3. Approach Movement Trend (APPROACHING / RECEDING / STABLE / UNKNOWN)
  4. Temporal Detection Persistence (consecutive frames confirmed)
  5. Herd / Group Size (count of simultaneously tracked elephants)
  6. Detection Confidence (sustained model probability 0.0 – 1.0)
  7. Dwell Duration within Risk Area (seconds)
  8. Recent Event History (incident context without unbounded feedback)

Safety Safeguards & Evidence Requirements:
  - Factor-level point caps and total score clamping to [0, 100].
  - Deterministic: identical inputs guarantee identical assessment.
  - Single-frame false-positive suppression: unconfirmed detections cannot
    produce CRITICAL or HIGH risk regardless of location.
  - CRITICAL requirements: verified persistence, minimum confidence (>= 0.70),
    and spatial proximity to protected boundary.
  - Evidence-based escalation: risk can elevate when conditions worsen, and
    de-escalate smoothly when targets recede or leave.

⚠️ SCIENTIFIC & ETHICAL DISCLAIMER:
  - This is an engineering decision-support prototype, not a certified wildlife safety system.
  - The scoring rules are deterministic heuristics, NOT a validated biological behavior model.
  - Camera location does NOT equal elephant location; monocular cameras cannot produce
    real-world GPS without calibrated depth/ranging sensors.
  - Image-space movement (UP/DOWN/LEFT/RIGHT) is NOT compass direction.
=============================================================================
"""

import math
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Import Zone Types from Geofencing
from ai.geofence import (
    ZONE_TYPE_MONITORING,
    ZONE_TYPE_OUTSIDE,
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
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

# Event Lifecycle States
EVENT_STATE_IDLE = "IDLE"
EVENT_STATE_DETECTED = "DETECTED"
EVENT_STATE_CONFIRMED = "CONFIRMED"
EVENT_STATE_HIGH_RISK = "HIGH_RISK"
EVENT_STATE_CRITICAL = "CRITICAL"
EVENT_STATE_RESOLVED = "RESOLVED"

VALID_EVENT_STATES = [
    EVENT_STATE_IDLE,
    EVENT_STATE_DETECTED,
    EVENT_STATE_CONFIRMED,
    EVENT_STATE_HIGH_RISK,
    EVENT_STATE_CRITICAL,
    EVENT_STATE_RESOLVED,
]

# Numeric priority mapping for risk and alert levels
LEVEL_PRIORITY: Dict[str, int] = {
    RISK_LOW: 1,
    RISK_MEDIUM: 2,
    RISK_HIGH: 3,
    RISK_CRITICAL: 4,
}


# =============================================================================
# 📈 MOVEMENT TREND TRACKER
# =============================================================================


class MovementTrendTracker:
    """
    Maintains a temporal history of distance measurements from a tracked elephant
    to a sensitive protected zone, and computes whether the elephant is
    APPROACHING, RECEDING, or STABLE relative to the protected area.

    Note: This evaluates geographic distance changes over time (when simulated or
    real coordinates are available). It is distinct from 2D camera image-space movement.
    """

    def __init__(
        self,
        window_size: int = 4,
        stability_threshold_meters: float = 15.0,
    ):
        self.window_size = max(2, int(window_size))
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
# ⚖️ EXPLICIT THREAT ASSESSMENT MODEL
# =============================================================================


class ThreatAssessment:
    """
    Internal structured representation for threat assessment.
    Combines deterministic multi-signal scoring with explainable factual reasons.

    Provides full backward compatibility with the legacy RiskAssessment interface.
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
        reasons: Optional[List[str]] = None,
        detection_count: int = 1,
        tracked_objects: Optional[List[int]] = None,
        duration_seconds: float = 0.0,
        event_id: Optional[str] = None,
        is_escalated: bool = False,
        timestamp: Optional[str] = None,
        is_resolved: bool = False,
    ):
        self.score = int(score)
        self.level = str(level).upper()
        self.zone = str(zone).upper()
        self.distance_to_protected = float(distance_to_protected)
        self.trend = str(trend).upper()
        self.group_size = int(group_size)
        self.confidence = float(confidence)
        self.breakdown = dict(breakdown)
        self.contributing_factors = self.breakdown
        self.alert_recommended = bool(alert_recommended)
        self.reasons = list(reasons or [])
        self.detection_count = int(detection_count)
        self.tracked_objects = list(tracked_objects or [])
        self.duration_seconds = float(duration_seconds)
        self.event_id = event_id
        self.is_escalated = bool(is_escalated)
        self.is_resolved = bool(is_resolved)
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    # Convenience aliases for explicit threat assessment model
    @property
    def risk_score(self) -> int:
        return self.score

    @property
    def risk_level(self) -> str:
        return self.level

    @property
    def zone_status(self) -> str:
        return self.zone

    @property
    def movement_status(self) -> str:
        return self.trend

    @property
    def distance_to_protected_m(self) -> float:
        return self.distance_to_protected

    def to_dict(self) -> dict:
        """
        Serializes assessment into a dictionary containing both legacy and
        new Phase 7 fields for full compatibility.
        """
        dist_val = round(self.distance_to_protected, 1) if not math.isinf(self.distance_to_protected) else float("inf")
        return {
            "risk_score": self.score,
            "risk_level": self.level,
            "zone": self.zone,
            "zone_status": self.zone,
            "distance_to_protected_m": dist_val,
            "trend": self.trend,
            "movement_status": self.trend,
            "group_size": self.group_size,
            "confidence": round(self.confidence, 3),
            "alert_recommended": self.alert_recommended,
            "breakdown": self.breakdown,
            "contributing_factors": self.contributing_factors,
            "reasons": list(self.reasons),
            "detection_count": self.detection_count,
            "tracked_objects": list(self.tracked_objects),
            "duration_seconds": round(self.duration_seconds, 2),
            "event_id": self.event_id,
            "is_escalated": self.is_escalated,
            "is_resolved": self.is_resolved,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        reasons_summary = f", reasons={len(self.reasons)}" if self.reasons else ""
        return (
            f"ThreatAssessment(level='{self.level}', score={self.score}/100, "
            f"zone='{self.zone}', dist={self.distance_to_protected:.0f}m, "
            f"trend='{self.trend}', group={self.group_size}{reasons_summary})"
        )


# Backward-compatible alias for existing code and tests
RiskAssessment = ThreatAssessment


# =============================================================================
# 🧠 RULE-BASED RISK ENGINE 2.0
# =============================================================================


class RiskEngine:
    """
    Transparent, deterministic rule-based threat evaluation engine (Risk Engine 2.0).

    Combines multiple evidence signals:
      - Zone membership
      - Proximity distance
      - Movement approach trend
      - Group size (herd count)
      - Sustained detection confidence
      - Temporal persistence (frames)
      - Duration in risk area (seconds)
      - Recent incident context

    Applies strict evidence safeguards:
      - Point contributions are strictly bounded per signal.
      - Total score is clamped to [0, 100].
      - Single-frame detections are capped below critical/high to suppress false alarms.
      - Critical risk requires verified persistence, confidence, and spatial proximity.
      - Every produced assessment includes human-readable, factual reasons explaining why.
    """

    def __init__(
        self,
        low_max: int = 24,
        medium_max: int = 49,
        high_max: int = 74,
        alert_on_levels: Optional[List[str]] = None,
        confirmation_frames: int = 5,
        critical_min_confidence: float = 0.70,
        critical_max_distance: float = 300.0,
        single_frame_risk_cap: str = RISK_MEDIUM,
        critical_requires_persistence: bool = True,
        herd_enabled: bool = True,
        herd_threshold: int = 3,
        movement_enabled: bool = True,
        # Maximum contribution per factor
        max_zone_points: int = 40,
        max_proximity_points: int = 25,
        max_trend_points: int = 20,
        max_persistence_points: int = 20,
        max_group_points: int = 10,
        max_confidence_points: int = 10,
        max_duration_points: int = 10,
        max_history_points: int = 5,
    ):
        self.low_max = int(low_max)
        self.medium_max = int(medium_max)
        self.high_max = int(high_max)
        self.alert_on_levels = set(alert_on_levels or [RISK_HIGH, RISK_CRITICAL])

        self.confirmation_frames = max(1, int(confirmation_frames))
        self.critical_min_confidence = float(critical_min_confidence)
        self.critical_max_distance = float(critical_max_distance)
        self.single_frame_risk_cap = str(single_frame_risk_cap).upper()
        self.critical_requires_persistence = bool(critical_requires_persistence)

        self.herd_enabled = bool(herd_enabled)
        self.herd_threshold = max(2, int(herd_threshold))
        self.movement_enabled = bool(movement_enabled)

        # Factor point limits
        self.max_zone_points = int(max_zone_points)
        self.max_proximity_points = int(max_proximity_points)
        self.max_trend_points = int(max_trend_points)
        self.max_persistence_points = int(max_persistence_points)
        self.max_group_points = int(max_group_points)
        self.max_confidence_points = int(max_confidence_points)
        self.max_duration_points = int(max_duration_points)
        self.max_history_points = int(max_history_points)

    def evaluate(
        self,
        zone: str,
        distance_to_protected: float,
        trend: str = TREND_UNKNOWN,
        group_size: int = 1,
        confidence: float = 1.0,
        *,
        persistence_frames: Optional[int] = None,
        duration_seconds: float = 0.0,
        tracked_objects: Optional[List[Any]] = None,
        recent_alert_count: int = 0,
        image_movement: Optional[str] = None,
        camera_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        event_id: Optional[str] = None,
        is_escalated: bool = False,
    ) -> ThreatAssessment:
        """
        Computes a deterministic numerical threat score (0–100), maps it to
        an actionable risk level (LOW, MEDIUM, HIGH, CRITICAL), enforces safety
        safeguards, and generates explainable reasons.

        Backward compatibility:
          Calling with the 5 standard positional arguments (zone, distance, trend,
          group_size, confidence) produces scores identical to Risk Engine 1.0.
        """
        zone_clean = str(zone).upper().strip() if zone else ZONE_TYPE_OUTSIDE
        trend_clean = str(trend).upper().strip() if trend else TREND_UNKNOWN

        try:
            dist = float(distance_to_protected)
            if not math.isinf(dist):
                dist = max(0.0, dist)
        except (ValueError, TypeError):
            dist = float("inf")

        group = max(0, int(group_size)) if group_size is not None else 1
        conf = min(1.0, max(0.0, float(confidence))) if confidence is not None else 1.0
        persistence = max(0, int(persistence_frames)) if persistence_frames is not None else None
        dur = max(0.0, float(duration_seconds)) if duration_seconds is not None else 0.0
        recents = max(0, int(recent_alert_count)) if recent_alert_count is not None else 0

        # Extract tracked object integer IDs
        t_ids: List[int] = []
        if tracked_objects:
            for obj in tracked_objects:
                if isinstance(obj, int):
                    t_ids.append(obj)
                elif isinstance(obj, dict):
                    tid = obj.get("id", obj.get("track_id"))
                    if tid is not None:
                        t_ids.append(int(tid))
                elif hasattr(obj, "track_id"):
                    t_ids.append(int(obj.track_id))

        breakdown: Dict[str, int] = {}
        factor_reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Zone Severity Score (Max: 40 points)
        # ---------------------------------------------------------------------
        if zone_clean in (ZONE_TYPE_PROTECTED, "PROTECTED", "HUMAN_ZONE"):
            zone_pts = min(40, self.max_zone_points)
            zone_reason = "Target located inside protected human settlement perimeter (highest priority zone)."
        elif zone_clean in (ZONE_TYPE_WARNING, "WARNING", "BUFFER"):
            zone_pts = min(20, self.max_zone_points)
            zone_reason = "Target located in intermediate buffer warning corridor."
        elif zone_clean in (ZONE_TYPE_MONITORING, "MONITORING", "FOREST"):
            zone_pts = min(5, self.max_zone_points)
            zone_reason = "Target located in natural forest monitoring perimeter."
        else:
            zone_pts = 0
            zone_reason = "Target is outside designated monitoring zones."

        breakdown["zone_points"] = zone_pts
        factor_reasons.append(zone_reason)

        # ---------------------------------------------------------------------
        # 2. Proximity to Protected Area (Max: 25 points)
        # ---------------------------------------------------------------------
        if dist <= 0.0 or zone_clean in (ZONE_TYPE_PROTECTED, "PROTECTED", "HUMAN_ZONE"):
            dist_pts = min(25, self.max_proximity_points)
            dist_reason = "Proximity: Zero distance to protected boundary (inside sensitive perimeter)."
        elif dist <= 200.0:
            dist_pts = min(20, self.max_proximity_points)
            dist_reason = f"Proximity: Immediate proximity ({dist:.0f}m) to protected boundary (<200m)."
        elif dist <= 500.0:
            dist_pts = min(15, self.max_proximity_points)
            dist_reason = f"Proximity: Close proximity ({dist:.0f}m) to protected boundary (<500m)."
        elif dist <= 1000.0:
            dist_pts = min(5, self.max_proximity_points)
            dist_reason = f"Proximity: Intermediate distance ({dist:.0f}m) to protected boundary (<1000m)."
        else:
            dist_pts = 0
            dist_reason = f"Proximity: Substantial distance ({dist:.0f}m) from protected area."

        breakdown["proximity_points"] = dist_pts
        factor_reasons.append(dist_reason)

        # ---------------------------------------------------------------------
        # 3. Approach Movement Trend (Max: 20 points)
        # ---------------------------------------------------------------------
        if not self.movement_enabled:
            trend_pts = 0
            trend_reason = "Movement: Trend scoring disabled by configuration."
        elif trend_clean == TREND_APPROACHING:
            trend_pts = min(20, self.max_trend_points)
            trend_reason = "Movement: Tracked movement is consistently APPROACHING protected boundary."
        elif trend_clean == TREND_RECEDING:
            trend_pts = 0
            trend_reason = "Movement: Tracked movement is RECEDING away from protected boundary."
        elif trend_clean in (TREND_STABLE, TREND_UNKNOWN):
            trend_pts = min(5, self.max_trend_points)
            if trend_clean == TREND_STABLE:
                trend_reason = "Movement: Tracked position is STABLE (stationary relative to protected boundary)."
            else:
                trend_reason = "Movement: Approach trend is UNKNOWN (insufficient trajectory history)."
        else:
            trend_pts = min(5, self.max_trend_points)
            trend_reason = f"Movement: Trend classified as {trend_clean}."

        breakdown["trend_points"] = trend_pts
        factor_reasons.append(trend_reason)

        # ---------------------------------------------------------------------
        # 4. Group Size / Herd Count (Max: 10 points)
        # ---------------------------------------------------------------------
        if not self.herd_enabled:
            group_pts = min(2, self.max_group_points) if group >= 1 else 0
            group_reason = f"Group size: {group} elephant(s) (herd scoring disabled)."
        elif group >= 4:
            group_pts = min(10, self.max_group_points)
            group_reason = f"Group size: Large herd detected ({group} elephants tracked simultaneously)."
        elif group >= 2:
            group_pts = min(5, self.max_group_points)
            group_reason = f"Group size: Multiple elephants detected ({group} tracked simultaneously)."
        elif group == 1:
            group_pts = min(2, self.max_group_points)
            group_reason = "Group size: Solitary elephant detected."
        else:
            group_pts = 0
            group_reason = "Group size: No active elephant tracks."

        breakdown["group_points"] = group_pts
        factor_reasons.append(group_reason)

        # ---------------------------------------------------------------------
        # 5. Detection Confidence (Max: 10 points)
        # ---------------------------------------------------------------------
        if conf >= 0.90:
            conf_pts = min(10, self.max_confidence_points)
            conf_reason = f"Confidence: High detection confidence sustained ({conf:.1%})."
        elif conf >= 0.80:
            conf_pts = min(7, self.max_confidence_points)
            conf_reason = f"Confidence: Moderate-high detection confidence ({conf:.1%})."
        elif conf >= 0.70:
            conf_pts = min(5, self.max_confidence_points)
            conf_reason = f"Confidence: Acceptable detection confidence ({conf:.1%})."
        else:
            conf_pts = 0
            conf_reason = f"Confidence: Low detection confidence ({conf:.1%})."

        breakdown["confidence_points"] = conf_pts
        factor_reasons.append(conf_reason)

        # ---------------------------------------------------------------------
        # 6. Temporal Persistence (Max: 20 points, Optional)
        # ---------------------------------------------------------------------
        if persistence is not None:
            if persistence >= 10:
                persist_pts = min(15, self.max_persistence_points)
                persist_reason = f"Persistence: Strong temporal persistence across {persistence} consecutive frames."
            elif persistence >= self.confirmation_frames:
                persist_pts = min(10, self.max_persistence_points)
                persist_reason = (
                    f"Persistence: Confirmed temporal presence sustained over {persistence} consecutive frames."
                )
            elif persistence >= 3:
                persist_pts = min(5, self.max_persistence_points)
                persist_reason = (
                    f"Persistence: Emerging candidate detection ({persistence}/{self.confirmation_frames} frames)."
                )
            else:
                persist_pts = 0
                persist_reason = (
                    f"Persistence: Single-frame candidate ({persistence} frame): unconfirmed temporal evidence."
                )

            breakdown["persistence_points"] = persist_pts
            factor_reasons.append(persist_reason)

        # ---------------------------------------------------------------------
        # 7. Duration within Risk Area (Max: 10 points, Optional)
        # ---------------------------------------------------------------------
        if (
            zone_clean in (ZONE_TYPE_PROTECTED, ZONE_TYPE_WARNING, "PROTECTED", "WARNING") or dist <= 500.0
        ) and dur > 0:
            if dur >= 30.0:
                dur_pts = min(10, self.max_duration_points)
                dur_reason = f"Duration: Prolonged dwell time ({dur:.1f}s) within elevated risk zone."
            elif dur >= 10.0:
                dur_pts = min(5, self.max_duration_points)
                dur_reason = f"Duration: Sustained dwell time ({dur:.1f}s) in monitored corridor."
            else:
                dur_pts = 0
                dur_reason = None

            if dur_pts > 0:
                breakdown["duration_points"] = dur_pts
                if dur_reason:
                    factor_reasons.append(dur_reason)

        # ---------------------------------------------------------------------
        # 8. Alert History Context (Max: 5 points, Optional)
        # ---------------------------------------------------------------------
        if recents >= 3:
            hist_pts = min(5, self.max_history_points)
            hist_reason = f"History: Frequent recent activity ({recents} prior alerts in area)."
        elif recents >= 1:
            hist_pts = min(2, self.max_history_points)
            hist_reason = f"History: Recent activity recorded ({recents} prior alert in area)."
        else:
            hist_pts = 0
            hist_reason = None

        if hist_pts > 0:
            breakdown["history_points"] = hist_pts
            if hist_reason:
                factor_reasons.append(hist_reason)

        # ---------------------------------------------------------------------
        # Aggregate Raw Score
        # ---------------------------------------------------------------------
        raw_score = sum(breakdown.values())
        clamped_score = min(100, max(0, raw_score))

        # Map score to preliminary risk level
        if clamped_score <= self.low_max:
            preliminary_level = RISK_LOW
        elif clamped_score <= self.medium_max:
            preliminary_level = RISK_MEDIUM
        elif clamped_score <= self.high_max:
            preliminary_level = RISK_HIGH
        else:
            preliminary_level = RISK_CRITICAL

        # ---------------------------------------------------------------------
        # 🛡️ EVIDENCE SAFEGUARDS & CRITICAL SAFETY RULES
        # ---------------------------------------------------------------------
        final_level = preliminary_level
        final_score = clamped_score
        safeguard_reasons: List[str] = []

        # Safeguard 1: Single-frame / unconfirmed false-positive suppression
        if persistence is not None and persistence < self.confirmation_frames:
            if preliminary_level in (RISK_HIGH, RISK_CRITICAL):
                final_level = self.single_frame_risk_cap
                final_score = min(final_score, self.medium_max)
                safeguard_reasons.append(
                    f"Evidence safeguard: Unconfirmed detection ({persistence}/{self.confirmation_frames} frames) "
                    f"capped at {final_level} to prevent noisy false alarms."
                )

        # Safeguard 2: Critical evidence requirements
        if final_level == RISK_CRITICAL:
            missing_critical: List[str] = []

            if (
                self.critical_requires_persistence
                and persistence is not None
                and persistence < self.confirmation_frames
            ):
                missing_critical.append("insufficient temporal persistence")

            if conf < self.critical_min_confidence:
                missing_critical.append(f"confidence {conf:.1%} < required {self.critical_min_confidence:.1%}")

            if zone_clean not in (ZONE_TYPE_PROTECTED, "PROTECTED", "HUMAN_ZONE") and dist > self.critical_max_distance:
                missing_critical.append(f"distance {dist:.0f}m > critical perimeter {self.critical_max_distance:.0f}m")

            if missing_critical:
                final_level = RISK_HIGH
                final_score = min(final_score, self.high_max)
                safeguard_reasons.append(
                    f"Evidence safeguard: Critical risk downgraded to HIGH due to: {', '.join(missing_critical)}."
                )

        # Prepend safeguard reasons to explanatory list
        reasons = safeguard_reasons + factor_reasons

        # Determine alert recommendation
        alert_recommended = final_level in self.alert_on_levels
        if persistence is not None and persistence < self.confirmation_frames:
            alert_recommended = False

        return ThreatAssessment(
            score=final_score,
            level=final_level,
            zone=zone_clean,
            distance_to_protected=dist,
            trend=trend_clean,
            group_size=group,
            confidence=conf,
            breakdown=breakdown,
            alert_recommended=alert_recommended,
            reasons=reasons,
            detection_count=persistence if persistence is not None else 1,
            tracked_objects=t_ids,
            duration_seconds=dur,
            event_id=event_id,
            is_escalated=is_escalated,
            timestamp=timestamp,
        )


# =============================================================================
# 🚦 ALERT POLICY (DECOUPLED FROM CORE RISK ENGINE)
# =============================================================================


class AlertDecision:
    """
    Structured outcome of an alert policy evaluation.
    Explains whether, how, and why an alert should be dispatched.
    """

    def __init__(
        self,
        should_dispatch: bool,
        alert_level: str,
        reason: str,
        is_escalation: bool = False,
        cooldown_bypassed: bool = False,
        delivery_channels: Optional[List[str]] = None,
    ):
        self.should_dispatch = bool(should_dispatch)
        self.alert_level = str(alert_level).upper()
        self.reason = str(reason)
        self.is_escalation = bool(is_escalation)
        self.cooldown_bypassed = bool(cooldown_bypassed)
        self.delivery_channels = list(delivery_channels or ["log"])

    def to_dict(self) -> dict:
        return {
            "should_dispatch": self.should_dispatch,
            "alert_level": self.alert_level,
            "reason": self.reason,
            "is_escalation": self.is_escalation,
            "cooldown_bypassed": self.cooldown_bypassed,
            "delivery_channels": self.delivery_channels,
        }

    def __repr__(self) -> str:
        return (
            f"AlertDecision(dispatch={self.should_dispatch}, level='{self.alert_level}', "
            f"escalation={self.is_escalation}, channels={self.delivery_channels})"
        )


class AlertPolicy:
    """
    Decoupled policy layer that translates a ThreatAssessment and active event context
    into an alert dispatch decision.

    Guarantees:
      - LOW risk -> Log only (no alerts)
      - MEDIUM risk -> Local log and HUD warning (no remote alert)
      - HIGH / CRITICAL risk -> Alert dispatch if persistence and cooldown satisfied
      - Evidence-based Escalation -> Bypasses standard cooldown if risk jumps (e.g. HIGH -> CRITICAL)
    """

    def __init__(
        self,
        alert_on_levels: Optional[List[str]] = None,
        cooldown_seconds: float = 30.0,
        allow_escalation_bypass: bool = True,
        confirmation_frames: int = 5,
    ):
        self.alert_on_levels = set(alert_on_levels or [RISK_HIGH, RISK_CRITICAL])
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.allow_escalation_bypass = bool(allow_escalation_bypass)
        self.confirmation_frames = max(1, int(confirmation_frames))

    def evaluate(
        self,
        assessment: ThreatAssessment,
        current_time: float,
        last_alert_time: float,
        last_alert_level: Optional[str] = None,
        is_escalated: bool = False,
    ) -> AlertDecision:
        """
        Evaluates whether an alert should be dispatched given the current threat assessment
        and temporal cooldown context.
        """
        level = assessment.level

        # Rule 1: Level not in configured alert levels
        if level not in self.alert_on_levels:
            if level == RISK_MEDIUM:
                return AlertDecision(
                    should_dispatch=False,
                    alert_level=level,
                    reason=f"Risk level {level}: Local HUD warning only, no remote dispatch.",
                    delivery_channels=["log", "hud"],
                )
            return AlertDecision(
                should_dispatch=False,
                alert_level=level,
                reason=f"Risk level {level} below alert trigger threshold ({', '.join(sorted(self.alert_on_levels))}).",
                delivery_channels=["log"],
            )

        # Rule 2: Persistence confirmation requirement
        if assessment.detection_count < self.confirmation_frames:
            return AlertDecision(
                should_dispatch=False,
                alert_level=level,
                reason=(
                    f"Unconfirmed temporal persistence ({assessment.detection_count}/{self.confirmation_frames} frames): "
                    "Alert withheld."
                ),
                delivery_channels=["log"],
            )

        # Rule 3: Escalation check
        current_priority = LEVEL_PRIORITY.get(level, 1)
        last_priority = LEVEL_PRIORITY.get(str(last_alert_level).upper(), 0) if last_alert_level else 0
        escalated = is_escalated or (last_priority > 0 and current_priority > last_priority)

        # Rule 4: Cooldown evaluation
        time_since_last = current_time - last_alert_time
        in_cooldown = time_since_last < self.cooldown_seconds

        if in_cooldown:
            if escalated and self.allow_escalation_bypass:
                return AlertDecision(
                    should_dispatch=True,
                    alert_level=level,
                    reason=(
                        f"Cooldown bypassed due to evidence-based risk escalation ({last_alert_level} -> {level})."
                    ),
                    is_escalation=True,
                    cooldown_bypassed=True,
                    delivery_channels=["log", "hud", "telegram"],
                )
            else:
                remaining = self.cooldown_seconds - time_since_last
                return AlertDecision(
                    should_dispatch=False,
                    alert_level=level,
                    reason=f"Alert suppressed by cooldown ({remaining:.1f}s remaining).",
                    delivery_channels=["log", "hud"],
                )

        # Rule 5: Confirmed dispatch
        return AlertDecision(
            should_dispatch=True,
            alert_level=level,
            reason=f"Confirmed {level} threat (score: {assessment.score}/100): Alert dispatch approved.",
            is_escalation=escalated,
            cooldown_bypassed=False,
            delivery_channels=["log", "hud", "telegram"],
        )


# =============================================================================
# 🔄 STATEFUL EVENT LIFECYCLE MANAGER
# =============================================================================


class EventTransition:
    """Holds transition metadata emitted by the EventLifecycleManager."""

    def __init__(
        self,
        event_id: Optional[str],
        state: str,
        transition_type: str,
        duration: float,
        current_risk_level: str,
        max_risk_level: str,
        is_escalated: bool = False,
        consecutive_detections: int = 0,
        resolved_event_id: Optional[str] = None,
    ):
        self.event_id = event_id
        self.state = state
        self.transition_type = transition_type
        self.duration = float(duration)
        self.current_risk_level = current_risk_level
        self.max_risk_level = max_risk_level
        self.is_escalated = bool(is_escalated)
        self.consecutive_detections = int(consecutive_detections)
        self.resolved_event_id = resolved_event_id

    def __repr__(self) -> str:
        return (
            f"EventTransition(state='{self.state}', type='{self.transition_type}', "
            f"id='{self.event_id}', dur={self.duration:.1f}s, level='{self.current_risk_level}')"
        )


class EventLifecycleManager:
    """
    Lightweight, in-memory state machine that manages the lifecycle of an elephant encounter:

      IDLE
       ↓ (elephant detected)
      DETECTED (candidate unconfirmed)
       ↓ (sustained across confirmation_frames)
      CONFIRMED (LOW/MEDIUM risk)
       ↓ (elevated threat)
      HIGH_RISK / CRITICAL
       ↓ (conditions improve)
      DE-ESCALATED
       ↓ (elephant leaves for resolution threshold)
      RESOLVED (active incident closed, stats archived)
    """

    def __init__(
        self,
        confirmation_frames: int = 5,
        resolution_frames: int = 30,
        resolution_seconds: float = 2.0,
    ):
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.resolution_frames = max(1, int(resolution_frames))
        self.resolution_seconds = max(0.1, float(resolution_seconds))

        self.state: str = EVENT_STATE_IDLE
        self.active_event_id: Optional[str] = None
        self.start_time: float = 0.0
        self.last_seen_time: float = 0.0
        self.consecutive_detections: int = 0
        self.consecutive_misses: int = 0

        self.current_risk_level: str = RISK_LOW
        self.max_risk_level: str = RISK_LOW
        self.last_alert_level: Optional[str] = None
        self.last_alert_time: float = 0.0
        self.is_escalated: bool = False

    def update(
        self,
        elephant_detected: bool,
        current_time: Optional[float] = None,
        risk_assessment: Optional[ThreatAssessment] = None,
    ) -> EventTransition:
        """
        Advances the event state machine given the detection presence in the current frame.
        """
        now = float(current_time) if current_time is not None else datetime.now(timezone.utc).timestamp()
        transition_type = "NO_CHANGE"
        resolved_id = None

        if elephant_detected:
            self.consecutive_misses = 0
            self.consecutive_detections += 1
            self.last_seen_time = now

            if self.active_event_id is None:
                # Brand new event
                self.active_event_id = f"zogan-{uuid.uuid4().hex[:12]}"
                self.start_time = now
                self.state = EVENT_STATE_DETECTED
                self.current_risk_level = RISK_LOW
                self.max_risk_level = RISK_LOW
                self.last_alert_level = None
                self.is_escalated = False
                transition_type = "NEW_EVENT"
            else:
                # Continuing event
                if self.state == EVENT_STATE_DETECTED and self.consecutive_detections >= self.confirmation_frames:
                    self.state = EVENT_STATE_CONFIRMED
                    transition_type = "CONFIRMED"
                else:
                    transition_type = "PERSISTING"

            if risk_assessment:
                new_level = risk_assessment.level
                curr_prio = LEVEL_PRIORITY.get(new_level, 1)
                max_prio = LEVEL_PRIORITY.get(self.max_risk_level, 1)
                last_alert_prio = LEVEL_PRIORITY.get(self.last_alert_level or "", 0)

                # Check escalation against previous alert level
                if last_alert_prio > 0 and curr_prio > last_alert_prio:
                    self.is_escalated = True
                    transition_type = "ESCALATED"
                else:
                    self.is_escalated = False

                self.current_risk_level = new_level
                if curr_prio > max_prio:
                    self.max_risk_level = new_level

                # Update state label based on risk
                if self.consecutive_detections >= self.confirmation_frames:
                    if new_level == RISK_CRITICAL:
                        self.state = EVENT_STATE_CRITICAL
                    elif new_level == RISK_HIGH:
                        self.state = EVENT_STATE_HIGH_RISK
                    else:
                        self.state = EVENT_STATE_CONFIRMED

        else:
            # Elephant not detected in frame
            if self.active_event_id is not None:
                self.consecutive_misses += 1
                self.consecutive_detections = 0
                time_lost = now - self.last_seen_time

                # Check if resolution conditions are satisfied
                if self.consecutive_misses >= self.resolution_frames or time_lost >= self.resolution_seconds:
                    resolved_id = self.active_event_id
                    duration = self.last_seen_time - self.start_time
                    transition_type = "RESOLVED"
                    self.state = EVENT_STATE_RESOLVED

                    # Reset internal state to IDLE for next encounter
                    trans = EventTransition(
                        event_id=resolved_id,
                        state=EVENT_STATE_RESOLVED,
                        transition_type="RESOLVED",
                        duration=max(0.0, duration),
                        current_risk_level=self.current_risk_level,
                        max_risk_level=self.max_risk_level,
                        is_escalated=False,
                        consecutive_detections=0,
                        resolved_event_id=resolved_id,
                    )
                    self._reset_to_idle()
                    return trans
                else:
                    transition_type = "GRACE_PERIOD"
            else:
                self.state = EVENT_STATE_IDLE
                transition_type = "IDLE"

        duration = (now - self.start_time) if self.active_event_id else 0.0
        return EventTransition(
            event_id=self.active_event_id,
            state=self.state,
            transition_type=transition_type,
            duration=max(0.0, duration),
            current_risk_level=self.current_risk_level,
            max_risk_level=self.max_risk_level,
            is_escalated=self.is_escalated,
            consecutive_detections=self.consecutive_detections,
            resolved_event_id=resolved_id,
        )

    def record_alert(self, alert_level: str, alert_time: float) -> None:
        """Records that an alert was dispatched at the specified level and timestamp."""
        self.last_alert_level = str(alert_level).upper()
        self.last_alert_time = float(alert_time)
        self.is_escalated = False

    def get_duration(self, current_time: Optional[float] = None) -> float:
        """Returns the elapsed duration of the currently active event in seconds."""
        if not self.active_event_id or self.start_time <= 0:
            return 0.0
        now = float(current_time) if current_time is not None else datetime.now(timezone.utc).timestamp()
        return max(0.0, now - self.start_time)

    def _reset_to_idle(self) -> None:
        self.state = EVENT_STATE_IDLE
        self.active_event_id = None
        self.start_time = 0.0
        self.last_seen_time = 0.0
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.current_risk_level = RISK_LOW
        self.max_risk_level = RISK_LOW
        self.last_alert_level = None
        self.last_alert_time = 0.0
        self.is_escalated = False

    def reset(self) -> None:
        """Manually forces a reset to clean IDLE state."""
        self._reset_to_idle()
